"""Exact block AC and LLT checkpointing at its native latent attention boundary.

LAC inputs are projected queries Q_r, the original shared KV latent C, and the
folded output weight. It recomputes attention and the folded output projection.
Query formation, residual and MLP paths remain unchanged outside this region.
No codec, additional trainable parameters, or approximate reconstruction.
"""
from torch.utils.checkpoint import checkpoint
from tensor_model import BackendTransformer


class CheckpointTransformer(BackendTransformer):
    def latent_region(self,q,latent,output_weight):
        a=self.attend(q,latent,latent).transpose(1,2).reshape(*q.shape[:1],q.shape[2],self.c.heads*self.c.rank)
        return self.linear(a,output_weight)

    def block(self,state,idx,policy,*shared):
        c,d=self.c,self.c.width//self.c.heads
        block,z=self.blocks[idx],self.normalize(state)
        if c.kind=='naive':
            q,k,v=[self.heads(self.linear(z,w.weight),d) for w in (block.q,block.k,block.v)]
            a=self.attend(q,k,v).transpose(1,2).reshape(*z.shape[:2],c.width)
            delta=self.linear(a,block.o.weight)
        else:
            latent,fq,fo=shared
            q=self.heads(self.linear(z,fq),c.rank)
            if policy=='lac':
                delta=checkpoint(self.latent_region,q,latent,fo,use_reentrant=False,preserve_rng_state=False)
            else:
                delta=self.latent_region(q,latent,fo)
        state=self.add(state,delta)
        return self.add(state,self.linear(self.activate(self.linear(self.normalize(state),block.w1.weight)),block.w2.weight))

    def forward(self,tokens,policy='none',return_cache=False,return_hidden=False,last_logits=False):
        if policy=='none':
            return super().forward(tokens,policy,return_cache,return_hidden,last_logits)
        if policy not in ('ac','lac') or return_cache:
            raise ValueError('unsupported checkpoint/cache policy')
        if policy=='lac' and self.c.kind!='llt':
            raise ValueError('LAC requires an existing LLT latent boundary; not applicable to conventional controls')
        c=self.c
        x=self.initial(tokens)
        latents=[] if c.kind=='naive' else [self.linear(self.normalize(x),down.weight).unsqueeze(1).contiguous() for down in self.down]
        folds=[] if c.kind=='naive' else [self.folds(b) for b in self.blocks]
        for loop in range(c.loops):
            for layer in range(c.layers):
                idx=loop*c.layers+layer if c.untied else layer
                shared=() if c.kind=='naive' else (latents[idx if c.kind=='layerwise' else 0],*folds[idx])
                if policy=='ac':
                    fn=lambda state,*args,idx=idx:self.block(state,idx,'none',*args)
                    x=checkpoint(fn,x,*shared,use_reentrant=False,preserve_rng_state=False)
                else:
                    x=self.block(x,idx,policy,*shared)
        hidden=self.normalize(x)
        return hidden if return_hidden else self.linear(hidden[:,-1:,:] if last_logits else hidden,self.output.weight)
