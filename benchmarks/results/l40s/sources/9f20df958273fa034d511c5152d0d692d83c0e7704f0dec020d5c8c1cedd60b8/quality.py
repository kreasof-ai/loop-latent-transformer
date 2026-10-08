"""Matched-token small language and pointer-chasing learning experiments.

This is exploratory trained-quality evidence, not a large-model benchmark.
All variants receive the same batches within a seed. Validation selects the
checkpoint; a disjoint fixed test is evaluated once after selection.
"""
import argparse
import gc
import hashlib
import math
import statistics
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import OUT, digest, environment, memory, save, setup, timing
from model import Config, Transformer

DATA_URL='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def language_data():
    path=OUT/'data/tinyshakespeare.txt'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(DATA_URL,timeout=60) as response:
            path.write_bytes(response.read())
    text=path.read_text()
    alphabet=sorted(set(text));lookup={ch:i for i,ch in enumerate(alphabet)}
    all_tokens=torch.tensor([lookup[ch] for ch in text],dtype=torch.long)
    a,b=int(len(text)*.9),int(len(text)*.95)
    splits=[all_tokens[:a],all_tokens[a:b],all_tokens[b:]]
    meta=dict(url=DATA_URL,sha256=digest(path),characters=len(text),alphabet=''.join(alphabet),
              split='contiguous 90% train / 5% validation / 5% test',boundaries=[a,b],
              context=128,objective='next-character cross entropy on all tokens')
    return splits,meta


def lm_batch(data, batch, seq, generator):
    starts=torch.randint(len(data)-seq,(batch,),generator=generator)
    ix=starts[:,None]+torch.arange(seq+1)[None,:]
    chunk=data[ix].cuda()
    return chunk[:,:-1],chunk[:,1:]


def pointer_batch(batch,generator,nodes=16,max_hops=4,fixed_hops=None):
    # A random permutation gives a complete directed graph of disjoint cycles.
    # Randomize pair order so sequence positions do not identify a key.
    mapping=torch.rand(batch,nodes,generator=generator).argsort(-1)
    order=torch.rand(batch,nodes,generator=generator).argsort(-1)
    start=torch.randint(nodes,(batch,),generator=generator)
    hops=torch.randint(1,max_hops+1,(batch,),generator=generator) if fixed_hops is None else torch.full((batch,),fixed_hops)
    answer=start.clone()
    for t in range(max_hops):
        answer=torch.where(hops>t,mapping.gather(1,answer[:,None])[:,0],answer)
    pairs=torch.stack((order,mapping.gather(1,order)),dim=-1).flatten(1)
    tokens=torch.cat((pairs,torch.full((batch,1),nodes),start[:,None],(nodes+hops)[:,None]),dim=1)
    return tokens.cuda(),answer.cuda(),hops.cuda()


@torch.no_grad()
def evaluate(model,task,data=None,seed=2027):
    model.eval();loss_sum=0.;count=0
    correct={h:[0,0] for h in range(1,5)}
    with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if task=='language':
            seq=128
            starts=torch.arange(0,len(data)-seq,seq)
            for chunk in starts.split(32):
                ix=chunk[:,None]+torch.arange(seq+1)[None,:]
                tokens=data[ix].cuda()
                logits=model(tokens[:,:-1])
                loss=F.cross_entropy(logits.flatten(0,1).float(),tokens[:,1:].flatten(),reduction='sum')
                loss_sum+=loss.item();count+=tokens[:,1:].numel()
        else:
            generator=torch.Generator().manual_seed(seed)
            for hop in range(1,5):
                for _ in range(8):
                    tokens,targets,hops=pointer_batch(128,generator,fixed_hops=hop)
                    logits=model(tokens)[:,-1,:16]
                    loss_sum+=F.cross_entropy(logits.float(),targets,reduction='sum').item()
                    count+=targets.numel()
                    correct[hop][0]+=(logits.argmax(-1)==targets).sum().item()
                    correct[hop][1]+=targets.numel()
    model.train()
    result=dict(loss_nats=loss_sum/count,examples_or_tokens=count)
    if task=='language':
        result['bits_per_character']=result['loss_nats']/math.log(2)
        result['character_perplexity']=math.exp(result['loss_nats'])
    else:
        result['accuracy']=sum(v[0] for v in correct.values())/count
        result['accuracy_by_hops']={h:v[0]/v[1] for h,v in correct.items()}
    return result


def configurations(task):
    values=[('naive_t1','naive',1,32,False),('naive_t4','naive',4,32,False),
            ('naive_untied_t4','naive',4,32,True),('llt_t1_r32','llt',1,32,False),
            ('llt_t4_r16','llt',4,16,False),('llt_t4_r32','llt',4,32,False),
            ('llt_t4_r64','llt',4,64,False),('layerwise_t4_r32','layerwise',4,32,False)]
    return values


def run(a):
    setup()
    splits,meta=language_data() if a.task=='language' else (None,dict(
        task='random-permutation pointer chasing',nodes=16,hops=[1,2,3,4],
        sequence='16 shuffled directed key/value pairs, query marker, start node, hop count',
        chance_accuracy=1/16,training='online IID generated instances',
        validation_seed=2027,test_seed=4099,test_examples=4096,objective='answer cross entropy at final position'))
    steps=a.steps or (800 if a.task=='language' else 1000)
    batch=16 if a.task=='language' else 64
    report=dict(status='running',environment=environment(),dataset=meta,protocol=dict(
        seeds=a.seeds,steps=steps,batch=batch,width=128,layers=2,heads=4,
        optimizer='AdamW, weight_decay=0.01, foreach=False',lr=.001,
        schedule='20-step warmup then cosine decay to 0.1x',gradient_clip=1.,
        precision='FP32 parameters and optimizer, BF16 autocast, CUDA Flash SDPA',
        checkpoint_policy='none for every model',positions='learned absolute, identical for all models',
        comparison='matched optimizer steps/data/compute depth for T4 variants, parameter counts and time reported; untied control uses more parameters',
        selection='minimum validation loss across scheduled evaluation steps; test once after selection'),runs=[])
    variants=configurations(a.task)
    if a.variants: variants=[v for v in variants if v[0] in a.variants]
    if a.smoke: variants=[v for v in variants if v[0]=='llt_t4_r32'];steps=4;report['protocol']['steps']=steps
    for seed in a.seeds:
        # Rotate method order across seeds to reduce systematic order effects.
        rotation=seed%len(variants)
        for name,kind,loops,rank,untied in variants[rotation:]+variants[:rotation]:
            gc.collect();torch.cuda.empty_cache();setup(seed)
            c=Config(kind=kind,width=128,heads=4,layers=2,loops=loops,rank=rank,
                     vocab=len(meta['alphabet']) if splits else 21,max_seq=128 if splits else 35,untied=untied)
            model=Transformer(c).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False)
            generator=torch.Generator().manual_seed(seed+10000)
            history=[];best_loss=float('inf');best_state=None;best_step=0
            torch.cuda.synchronize();started=time.perf_counter();train_seconds=0.
            for step in range(1,steps+1):
                lr=.001*min(1.,step/20)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
                for group in opt.param_groups: group['lr']=lr
                if splits:
                    tokens,targets=lm_batch(splits[0],batch,128,generator)
                else:
                    tokens,targets,_=pointer_batch(batch,generator)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    logits=model(tokens)
                    loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten()) if splits else F.cross_entropy(logits[:,-1,:16].float(),targets)
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {name}/{seed}/{step}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step%max(1,steps//4)==0 or step==steps:
                    torch.cuda.synchronize();train_seconds+=time.perf_counter()-started
                    val=evaluate(model,a.task,splits[1] if splits else None)
                    history.append(dict(step=step,train_loss=loss.item(),validation=val))
                    if val['loss_nats']<best_loss:
                        best_loss=val['loss_nats'];best_step=step
                        best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                    print(a.task,name,'seed',seed,'step',step,'val',round(val['loss_nats'],4),
                          'acc',val.get('accuracy'),flush=True)
                    started=time.perf_counter()
            model.load_state_dict(best_state)
            test=evaluate(model,a.task,splits[2] if splits else None,seed=4099)
            checkpoint=OUT/'checkpoints'/a.task/f'{name}-seed{seed}.pt'
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(config=asdict(c),seed=seed,step=best_step,model=best_state,dataset=meta),checkpoint)
            row=dict(name=name,seed=seed,config=asdict(c),parameters=sum(p.numel() for p in model.parameters()),
                     training_seconds=train_seconds,trained_tokens=steps*batch*(128 if splits else 35),
                     best_validation_step=best_step,history=history,test=test,
                     checkpoint=str(checkpoint.relative_to(OUT)),checkpoint_sha256=digest(checkpoint))
            report['runs'].append(row);save(OUT/f'{a.task}.json',report)
            print('TEST',a.task,name,seed,test,round(train_seconds,1),'seconds',flush=True)
            del model,opt,best_state,loss,logits,tokens,targets
    report['status']='passed';save(OUT/f'{a.task}.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['language','reasoning'])
    p.add_argument('--steps',type=int)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--variants',nargs='+')
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
