"""Portable forward kernels for batched LLT experiments; no training claims."""
from vulkan_attention import emit, attention_source, query_source, output_source, merge_source


def matrix(m, n, k, dtype='float16'):
    if dtype=='float16':
        from tensor.compiler.webgpu_lowering import register_matmul_schedule
        body=register_matmul_schedule(m,k,n,'value',
            f'T.cast(w[(bx*32+i)*{k}+tile*32+j], "float32")',
            tile_m=16,tile_n=32,tile_k=32,threads=128,dot_width=4)
        return emit([('x',m*k,dtype),('w',n*k,dtype),('out',m*n,dtype)],body)
    return emit([('x',m*k,dtype),('w',n*k,dtype),('out',m*n,dtype)],f'''with T.Kernel(T.ceildiv({m},16), T.ceildiv({n},32), threads=128) as (by,bx):
    a = T.alloc_shared((16,32), "{dtype}")
    b = T.alloc_shared((32,32), "{dtype}")
    acc = T.alloc_fragment((16,32), "float32")
    T.clear(acc)
    for tile in T.serial({k//32}):
        for i,j in T.Parallel(16,32):
            a[i,j] = T.if_then_else(by*16+i<{m}, x[(by*16+i)*{k}+tile*32+j], 0)
        for i,j in T.Parallel(32,32):
            b[i,j] = T.if_then_else(bx*32+i<{n}, w[(bx*32+i)*{k}+tile*32+j], 0)
        T.gemm(a,b,acc,transpose_B=True)
    for i,j in T.Parallel(16,32):
        if by*16+i<{m} and bx*32+j<{n}:
            out[(by*16+i)*{n}+bx*32+j] = acc[i,j]''')


def attention(s,h,d,r,batch,dtype,partitions,explicit=False):
    width=d if explicit else r
    text=attention_source(s,h,width,d,explicit=explicit,partitions=partitions)
    text=text.replace(f'with T.Kernel({partitions}, {h}, threads=128) as (part, head):',
                      f'with T.Kernel({partitions}, {h*batch}, threads=128) as (part, bh):\n        head = bh % {h}\n        batch = bh // {h}')
    text=text.replace(f'q[head*{width}+j]',f'q[bh*{width}+j]')
    text=text.replace(f'(head*{partitions}+part)',f'(bh*{partitions}+part)')
    if explicit:
        text=text.replace('kv[(',f'kv[batch*{s*2*h*d}+(')
        text=text.replace(f'kv: T.Tensor(({s*2*h*d},)',f'kv: T.Tensor(({batch*s*2*h*d},)')
    else:
        text=text.replace('c[(',f'c[batch*{s*r}+(')
        text=text.replace(f'c: T.Tensor(({s*r},)',f'c: T.Tensor(({batch*s*r},)')
    for name,size in [('q',h*width),('out',h*partitions*width),('stats',h*partitions*2)]:
        text=text.replace(f'{name}: T.Tensor(({size},)',f'{name}: T.Tensor(({batch*size},)')
    return text.replace('"float16"',f'"{dtype}"')


def query(h,d,r,batch,dtype):
    text=query_source(h,d,r)
    text=text.replace(f'with T.Kernel({h}, threads=128) as head:',
                      f'with T.Kernel({h*batch}, threads=128) as bh:\n        head = bh % {h}')
    text=text.replace(f'q[head*{d}+k]',f'q[bh*{d}+k]').replace(f'out[head*{r}+j]',f'out[bh*{r}+j]')
    for name,size in [('q',h*d),('out',h*r)]:
        text=text.replace(f'{name}: T.Tensor(({size},)',f'{name}: T.Tensor(({batch*size},)')
    return text.replace('"float16"',f'"{dtype}"')


def output(h,d,r,batch,dtype):
    text=output_source(h,d,r)
    text=text.replace(f'with T.Kernel({h}, threads=128) as head:',
                      f'with T.Kernel({h*batch}, threads=128) as bh:\n        head = bh % {h}')
    text=text.replace(f'latent[head*{r}+k]',f'latent[bh*{r}+k]').replace(f'out[head*{d}+j]',f'out[bh*{d}+j]')
    for name,size in [('latent',h*r),('out',h*d)]:
        text=text.replace(f'{name}: T.Tensor(({size},)',f'{name}: T.Tensor(({batch*size},)')
    return text.replace('"float16"',f'"{dtype}"')


def fused(s,h,d,r,batch,dtype,partitions):
    return emit([('q',batch*h*d,'float32'),('c',batch*s*r,dtype),('w',2*h*d*r,dtype),
                 ('out',batch*h*partitions*d,'float32'),('stats',batch*h*partitions*2,'float32')],f'''with T.Kernel({partitions}, {batch*h}, threads=128) as (part,bh):
    head = bh % {h}
    batch = bh // {h}
    lhs = T.alloc_shared((16,32), "{dtype}")
    rhs = T.alloc_shared(({d},32), "{dtype}")
    expansion = T.alloc_fragment((16,{d}), "float32")
    key = T.alloc_shared((16,{d}), "{dtype}")
    value = T.alloc_shared((16,{d}), "{dtype}")
    dots = T.alloc_fragment((16,{d}), "float32")
    scores = T.alloc_fragment((16,), "float32")
    products = T.alloc_fragment((16,{d}), "float32")
    partial = T.alloc_fragment(({d},), "float32")
    result = T.alloc_fragment(({d},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    previous = T.alloc_fragment((1,), "float32")
    normalizer = T.alloc_fragment((1,), "float32")
    total = T.alloc_fragment((1,), "float32")
    T.fill(maximum, -T.infinity("float32"))
    T.clear(normalizer)
    T.clear(result)
    for tile in T.serial({s//partitions//16}):
        T.clear(expansion)
        for kk in T.serial({r//32}):
            for i,j in T.Parallel(16,32):
                lhs[i,j] = c[(batch*{s}+part*{s//partitions}+tile*16+i)*{r}+kk*32+j]
            for i,j in T.Parallel({d},32):
                rhs[i,j] = w[(head*{d}+i)*{r}+kk*32+j]
            T.gemm(lhs,rhs,expansion,transpose_B=True)
        T.copy(expansion,key)
        T.clear(expansion)
        for kk in T.serial({r//32}):
            for i,j in T.Parallel(16,32):
                lhs[i,j] = c[(batch*{s}+part*{s//partitions}+tile*16+i)*{r}+kk*32+j]
            for i,j in T.Parallel({d},32):
                rhs[i,j] = w[({h*d}+head*{d}+i)*{r}+kk*32+j]
            T.gemm(lhs,rhs,expansion,transpose_B=True)
        T.copy(expansion,value)
        for i,j in T.Parallel(16,{d}):
            dots[i,j] = q[bh*{d}+j]*T.cast(key[i,j], "float32")
        T.reduce_sum(dots,scores,dim=1)
        T.copy(maximum,previous)
        for i in T.Parallel(16):
            scores[i] *= {d**-.5}
        T.reduce_max(scores,maximum,dim=0,clear=False)
        for i in T.Parallel(16):
            scores[i] = T.exp(scores[i]-maximum[0])
        T.reduce_sum(scores,total,dim=0)
        for z in T.Parallel(1):
            normalizer[0] = normalizer[0]*T.exp(previous[0]-maximum[0])+total[0]
        for i,j in T.Parallel(16,{d}):
            products[i,j] = scores[i]*T.cast(value[i,j], "float32")
        T.reduce_sum(products,partial,dim=0)
        for j in T.Parallel({d}):
            result[j] = result[j]*T.exp(previous[0]-maximum[0])+partial[j]
        T.sync_threads()
    for j in T.Parallel({d}):
        out[(bh*{partitions}+part)*{d}+j] = result[j]
    for z in T.Parallel(1):
        stats[(bh*{partitions}+part)*2] = maximum[0]
        stats[(bh*{partitions}+part)*2+1] = normalizer[0]''')


def gemv(batch,k,n,dtype='float16'):
    from tensor.compiler.webgpu_lowering import partitioned_matmul_schedule
    # Input stays FP32 throughout the residual stream. Only weights/cache use
    # the selected storage precision. Shared reductions have deterministic owners.
    body=partitioned_matmul_schedule(batch,k,n,
        f'x[({{row}})*{k}+({{k}})]',
        f'T.cast(w[({{column}})*{k}+({{k}})], "float32")',
        tile_m=1,tile_n=16,threads=128,partitions=8,unroll=4,dot_width=4)
    return emit([('x',batch*k,'float32'),('w',n*k,dtype),('out',batch*n,'float32')],body)


def norm(batch,width):
    return emit([('x',batch*width,'float32'),('out',batch*width,'float32')],f'''with T.Kernel({batch}, threads=128) as row:
    square = T.alloc_fragment(({width},), "float32")
    total = T.alloc_fragment((1,), "float32")
    for j in T.Parallel({width}):
        square[j] = x[row*{width}+j]*x[row*{width}+j]
    T.reduce_sum(square,total,dim=0)
    for j in T.Parallel({width}):
        out[row*{width}+j] = x[row*{width}+j]*T.rsqrt(total[0]/{width}+1e-5)''')


def pointwise(size,operation):
    if operation=='copy':
        return emit([('x',size,'float32'),('out',size,'float32')],f'''with T.Kernel(T.ceildiv({size},128),threads=128) as block:
    for j in T.Parallel(128):
        if block*128+j<{size}:
            out[block*128+j] = x[block*128+j]''')
    if operation=='gelu':
        return emit([('x',size,'float32'),('out',size,'float32')],f'''with T.Kernel(T.ceildiv({size},128),threads=128) as block:
    for j in T.Parallel(128):
        if block*128+j<{size}:
            out[block*128+j] = .5*x[block*128+j]*(1+T.tanh(.7978845608028654*(x[block*128+j]+.044715*x[block*128+j]*x[block*128+j]*x[block*128+j])))''')
    return emit([('x',size,'float32'),('y',size,'float32'),('out',size,'float32')],f'''with T.Kernel(T.ceildiv({size},128),threads=128) as block:
    for j in T.Parallel(128):
        if block*128+j<{size}:
            out[block*128+j] = x[block*128+j]+y[block*128+j]''')


def append(batch,s,width,dtype):
    return emit([('x',batch*width,'float32'),('cache',batch*s*width,dtype)],f'''with T.Kernel({batch},threads=128) as row:
    for j in T.Parallel({width}):
        cache[(row*{s}+{s-1})*{width}+j] = x[row*{width}+j]''')
