"""Tensor CUDA attention: shared latent KV, causal prefill and cached decode.

The serial online-softmax loop follows Tensor's validated forward example.
Shared KV is addressed directly; no replication across heads is stored.
"""
import hashlib
import json
import os
from pathlib import Path

from common import OUT, TENSOR, digest


def make_attention(batch, heads, kv_heads, m, n, dim, causal, scale, bm=32, bn=64):
    import tilelang.language as T

    @T.prim_func
    def attention(q: T.Tensor((batch, heads, m, dim), 'float16'),
                  k: T.Tensor((batch, kv_heads, n, dim), 'float16'),
                  v: T.Tensor((batch, kv_heads, n, dim), 'float16'),
                  out: T.Tensor((batch, heads, m, dim), 'float16')):
        with T.Kernel(T.ceildiv(m, bm), heads, batch, threads=128) as (blk, h, b):
            query = T.alloc_shared((bm, dim), 'float16')
            key = T.alloc_shared((bn, dim), 'float16')
            value = T.alloc_shared((bn, dim), 'float16')
            probability = T.alloc_shared((bm, bn), 'float16')
            scores = T.alloc_fragment((bm, bn), 'float32')
            result = T.alloc_fragment((bm, dim), 'float32')
            maximum = T.alloc_fragment((bm,), 'float32')
            previous = T.alloc_fragment((bm,), 'float32')
            factor = T.alloc_fragment((bm,), 'float32')
            normalizer = T.alloc_fragment((bm,), 'float32')
            tile_sum = T.alloc_fragment((bm,), 'float32')
            kh = h // (heads // kv_heads)
            T.copy(q[b, h, blk * bm:(blk + 1) * bm, :], query)
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity('float32'))
            tiles = T.ceildiv(T.min(n, n - m + (blk + 1) * bm), bn) if causal else T.ceildiv(n, bn)
            for tile in T.serial(tiles):
                T.copy(k[b, kh, tile * bn:(tile + 1) * bn, :], key)
                T.gemm(query, key, scores, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.if_then_else(
                        (tile * bn + j < n) & ((not causal) | (tile * bn + j <= n - m + blk * bm + i)),
                        scores[i, j] * scale, -T.infinity('float32'))
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1, clear=True)
                for i in T.Parallel(bm):
                    maximum[i] = T.max(previous[i], maximum[i])
                    factor[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                T.reduce_sum(scores, tile_sum, dim=1, clear=True)
                for i in T.Parallel(bm):
                    normalizer[i] = normalizer[i] * factor[i] + tile_sum[i]
                for i, j in T.Parallel(bm, dim):
                    result[i, j] *= factor[i]
                T.copy(scores, probability)
                T.copy(v[b, kh, tile * bn:(tile + 1) * bn, :], value)
                T.gemm(probability, value, result, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, dim):
                result[i, j] /= normalizer[i]
            T.copy(result, out[b, h, blk * bm:(blk + 1) * bm, :])
    return attention


def make_decode(batch, heads, kv_heads, n, dim, scale, partitions=32, bn=64):
    import tilelang.language as T
    bm=32
    tiles=(n+bn-1)//bn
    per=(tiles+partitions-1)//partitions
    @T.prim_func
    def decode(q: T.Tensor((batch,heads,1,dim),'float16'),
               k: T.Tensor((batch,kv_heads,n,dim),'float16'),
               v: T.Tensor((batch,kv_heads,n,dim),'float16'),
               partial: T.Tensor((batch,heads,partitions,dim),'float32'),
               stats: T.Tensor((batch,heads,partitions,2),'float32')):
        with T.Kernel(partitions,heads,batch,threads=128) as (p,h,b):
            query=T.alloc_shared((bm,dim),'float16')
            key=T.alloc_shared((bn,dim),'float16')
            value=T.alloc_shared((bn,dim),'float16')
            probability=T.alloc_shared((bm,bn),'float16')
            scores=T.alloc_fragment((bm,bn),'float32')
            result=T.alloc_fragment((bm,dim),'float32')
            maximum=T.alloc_fragment((bm,),'float32')
            previous=T.alloc_fragment((bm,),'float32')
            factor=T.alloc_fragment((bm,),'float32')
            normalizer=T.alloc_fragment((bm,),'float32')
            tile_sum=T.alloc_fragment((bm,),'float32')
            # A direct scalar extraction from a reduced fragment constrains its
            # replicated layout incompatibly in TileLang 0.1.14. Stage through
            # shared memory before exporting the single valid query row.
            result_shared=T.alloc_shared((bm,dim),'float32')
            maximum_shared=T.alloc_shared((bm,),'float32')
            normalizer_shared=T.alloc_shared((bm,),'float32')
            kh=h//(heads//kv_heads)
            T.copy(q[b,h,0:bm,:],query)
            T.clear(result);T.clear(normalizer);T.fill(maximum,-T.infinity('float32'))
            for local in T.serial(T.max(0,T.min(per,tiles-p*per))):
                tile=p*per+local
                T.copy(k[b,kh,tile*bn:(tile+1)*bn,:],key)
                T.gemm(query,key,scores,transpose_B=True,clear_accum=True,policy=T.GemmWarpPolicy.FullRow)
                for i,j in T.Parallel(bm,bn):
                    scores[i,j]=T.if_then_else(tile*bn+j<n,scores[i,j]*scale,-T.infinity('float32'))
                T.copy(maximum,previous);T.reduce_max(scores,maximum,dim=1,clear=True)
                for i in T.Parallel(bm):
                    maximum[i]=T.max(previous[i],maximum[i]);factor[i]=T.exp(previous[i]-maximum[i])
                for i,j in T.Parallel(bm,bn): scores[i,j]=T.exp(scores[i,j]-maximum[i])
                T.reduce_sum(scores,tile_sum,dim=1,clear=True)
                for i in T.Parallel(bm): normalizer[i]=normalizer[i]*factor[i]+tile_sum[i]
                for i,j in T.Parallel(bm,dim): result[i,j]*=factor[i]
                T.copy(scores,probability);T.copy(v[b,kh,tile*bn:(tile+1)*bn,:],value)
                T.gemm(probability,value,result,policy=T.GemmWarpPolicy.FullRow)
            T.copy(result,result_shared)
            T.copy(maximum,maximum_shared)
            T.copy(normalizer,normalizer_shared)
            for j in T.Parallel(dim): partial[b,h,p,j]=result_shared[0,j]
            stats[b,h,p,0]=maximum_shared[0];stats[b,h,p,1]=normalizer_shared[0]
    return decode


def make_merge(batch,heads,dim,partitions=32):
    import tilelang.language as T
    @T.prim_func
    def merge(partial:T.Tensor((batch,heads,partitions,dim),'float32'),
              stats:T.Tensor((batch,heads,partitions,2),'float32'),
              out:T.Tensor((batch,heads,1,dim),'float16')):
        with T.Kernel(heads,batch,threads=128) as (h,b):
            maxima=T.alloc_fragment((partitions,),'float32')
            weights=T.alloc_fragment((partitions,),'float32')
            normal=T.alloc_fragment((partitions,),'float32')
            total=T.alloc_fragment((1,),'float32')
            largest=T.alloc_fragment((1,),'float32')
            values=T.alloc_fragment((partitions,dim),'float32')
            accum=T.alloc_fragment((dim,),'float32')
            for p in T.Parallel(partitions): maxima[p]=stats[b,h,p,0]
            T.reduce_max(maxima,largest,dim=0,clear=True)
            for p in T.Parallel(partitions):
                weights[p]=T.exp(maxima[p]-largest[0]);normal[p]=weights[p]*stats[b,h,p,1]
            T.reduce_sum(normal,total,dim=0,clear=True)
            for p,j in T.Parallel(partitions,dim): values[p,j]=weights[p]*partial[b,h,p,j]
            T.reduce_sum(values,accum,dim=0,clear=True)
            for j in T.Parallel(dim): out[b,h,0,j]=accum[j]/total[0]
    return merge


class Attention:
    def __init__(self, out=OUT, partitions=32, split_decode=True):
        import tensor_torch
        self.adapter = tensor_torch
        self.root = Path(out) / 'artifacts'
        self.root.mkdir(parents=True, exist_ok=True)
        self.loaded, self.artifacts = {}, []
        self.partitions,self.split_decode=partitions,split_decode
        self.source_hash=digest(__file__)
        os.environ.setdefault('TENSOR_NVRTC_HOME', str(TENSOR / 'build/nvrtc-12.9'))

    def build(self, factory, spec, outputs):
        import tensor
        key=(factory,spec)
        if key not in self.loaded:
            identity = hashlib.sha256((factory + repr(spec) + self.source_hash).encode()).hexdigest()[:24]
            source = self.root / (identity + '.py')
            source.write_text('import sys\nsys.path.insert(0, ' + repr(str(Path(__file__).parent)) + ')\n'
                              'from kernels import '+factory+'\ndef tensor_export():\n'
                              '    return {"kernel": '+factory+'(*' + repr(spec) + '), "outputs": '+repr(outputs)+'}\n')
            artifact = source.with_suffix('.tbin')
            if not artifact.exists(): tensor.build(source, artifact, cache_dir=self.root / 'cache')
            self.loaded[key] = self.adapter.load(artifact)
            self.artifacts.append(dict(factory=factory,spec=spec,path=str(artifact.relative_to(OUT.parent.parent.parent)),
                                       sha256=digest(artifact),source_sha256=digest(source)))
        return self.loaded[key]

    def kernel(self, q, k, causal, scale):
        b, h, m, d = q.shape
        bk, kh, n, dk = k.shape
        assert b == bk and d == dk and h % kh == 0
        assert d % 16 == 0 and m <= n
        spec = (b, h, kh, m, n, d, causal, scale)
        return self.build('make_attention',spec,['out'])

    def decode_kernels(self,q,k,scale):
        b,h,m,d=q.shape;kh,n=k.shape[1:3]
        first=self.build('make_decode',(b,h,kh,n,d,scale,self.partitions),['partial','stats'])
        second=self.build('make_merge',(b,h,d,self.partitions),['out'])
        return first,second

    def prepare(self,q,k,v,causal,scale):
        if q.shape[2]==1 and not causal and self.split_decode:
            first,second=self.decode_kernels(q,k,scale)
            a=first.prepare(q,k,v)
            b=second.prepare(*a.outputs)
            class Plan:
                outputs=b.outputs
                def __call__(self): a();b()
            return Plan()
        return self.kernel(q,k,causal,scale).prepare(q,k,v)

    def __call__(self, q, k, v, causal, scale):
        # The model passes zero-stride expanded head views during prefill.
        if k.stride(1) == 0:
            k = k[:, :1].contiguous()
        if v.stride(1) == 0:
            v = v[:, :1].contiguous()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if q.shape[2]==1 and not causal and self.split_decode:
            first,second=self.decode_kernels(q,k,scale)
            partial,stats=first(q,k,v)
            return second(partial,stats)
        return self.kernel(q, k, causal, scale)(q, k, v)
