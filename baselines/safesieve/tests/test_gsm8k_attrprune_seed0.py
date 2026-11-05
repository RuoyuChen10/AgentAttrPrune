"""Offline alignment, native GSM8K pruning, microbatching and resume checks."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import run_gsm8k_attrprune_seed0 as run
from experiments import attrprune_gsm8k_seed0_adapter as adapter
from AgentAttrPrune.prompt.gsm8k import WORKER_ROLES


@pytest.fixture
def alignment(monkeypatch):
    workers=[{'agent_id':a,'role':r,'node_index':i} for i,(a,r) in enumerate(WORKER_ROLES)]
    def samples(prefix,n):
        return [{'task':f'{prefix} question {i}','question':f'{prefix} question {i}','answer':str(i), 'benchmark':'gsm8k'} for i in range(n)]
    item={'workers':workers,'reviewer_agent_ids':{w['agent_id']:f'Ab{i:02}' for i,w in enumerate(workers)},
          'calibration_samples':samples('train',40),'evaluation_samples':samples('test',1319),
          'reference_run_config':{'model_name':'/model/Qwen3-8B','decoding':{'temperature':0.,'max_new_tokens':2048,'enable_thinking':False}},
          'reference_run':'/original/seed0','source_hashes':{},'fingerprint':'test-frozen-cohort'}
    monkeypatch.setattr(adapter,'_alignment',None)
    monkeypatch.setattr(run,'load_alignment',lambda _:copy.deepcopy(item))
    monkeypatch.setattr(run,'verify_sources',lambda _:None)
    return item


def fake_similarity(ctx):
    from SafeSieve.graph.prompt_similarity import _build_evaluation_prompt
    spec=ctx['config']['similarity']; pairs=[]; matrix=np.zeros((4,4))
    for i in range(4):
        for j in range(i+1,4):
            messages=[{'role':'user','content':_build_evaluation_prompt(spec['role_descriptions'][i],spec['role_descriptions'][j],'GSM8K',10.)}]
            pairs.append({'i':i,'j':j,'messages':messages,'response':'Score: 6.0','score':6.})
            matrix[i,j]=matrix[j,i]=.6
    return {'configuration':spec,'pairs':pairs,'matrix':matrix.tolist()}


@pytest.fixture
def ctx(tmp_path,alignment):
    args=run.parse_args(['--run-dir',str(tmp_path/'run'),'--similarity-json',str(tmp_path/'similarity.json')])
    ctx=run.prepare_context(args);ctx['args']=args
    run.write_json(args.similarity_json,fake_similarity(ctx))
    return ctx


class Tokenizer:
    pad_token_id=0; eos_token_id=2
    def __call__(self,text,return_tensors):
        ids=torch.tensor([[3]*(len(text)%11+2)])
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    def batch_decode(self,tokens,skip_special_tokens):return ['The answer is 1' for _ in tokens]


def make_llm():
    llm=object.__new__(run.HFChat);llm.model_name='local:/model/Qwen3-8B';llm.model_path='/model/Qwen3-8B'
    llm.logits_processors=[];llm.tokenizer=Tokenizer();llm._generation_lock=threading.Lock();llm.calls=[]
    llm.render_messages=lambda messages:'\n'.join(m['content'] for m in messages)
    def generate(input_ids,attention_mask,**kwargs):
        llm.calls.append({'batch_size':len(input_ids),'kwargs':kwargs})
        return SimpleNamespace(sequences=torch.cat([input_ids,input_ids.new_tensor([[9,2]]*len(input_ids))],dim=1))
    llm.model=SimpleNamespace(generation_config=SimpleNamespace(num_beams=1,min_length=0),device='cpu',generate=generate)
    def single(messages,*,max_tokens,temperature,num_comps):
        assert temperature==0 and num_comps==1
        response,event=llm._generate_batch([SimpleNamespace(messages=messages,max_tokens=max_tokens)])[0]
        observer=run.generation_observer.get()
        if observer:observer(event)
        return response
    llm._generate=single
    return llm


def factory(llm):return lambda _:llm


def test_budget_exact_original_gsm_protocol(ctx):
    c=ctx['config']
    assert c['training']['draw_plan']==list(range(40))
    assert c['training']['logical_batch_size']==4
    assert c['training']['logical_batch_count']==10
    assert c['training']['possible_pruning_batches']==[10]
    assert c['decoding']=={'temperature':0.,'max_new_tokens':2048,'enable_thinking':False}
    assert c['similarity']['mode']=='aligned_roles' and c['similarity']['old_role_descriptions_differ']
    assert c['backend_name']=='local:/model/Qwen3-8B'
    assert not ctx['directory'].exists()




def test_real_hf_microbatch_path_and_canonical_nodes(ctx):
    llm=make_llm();graph=run.build_graph(ctx,llm_factory=factory(llm))
    async def work():
        async with llm.async_batching(4):return await run.run_batch(graph,ctx['alignment']['calibration_samples'][:4])
    result=asyncio.run(work())
    assert max(c['batch_size'] for c in llm.calls)==4
    assert all(c['batch_size']<=4 and not c['kwargs']['do_sample'] and c['kwargs']['max_new_tokens']==2048 for c in llm.calls)
    for row in result:
        assert len(row['events'])==9
        assert {e['agent_id'] for e in row['events']}==set(ctx['config']['worker_ids'])|{'dense_reviewer'}
        assert all(e['generation_kwargs']['temperature']==0 for e in row['events'])
    assert not graph.spatial_adj_matrix.any()


def test_reviewer_after_node_removal_keeps_reference_alias_order(ctx):
    adapter.configure(ctx['alignment'],llm_factory=factory(make_llm()))
    node=adapter.AttrPruneGSM8KReviewer(llm_name='local:/model/Qwen3-8B')
    selected=ctx['alignment']['workers'][::2][::-1]
    info={w['agent_id']:{'role':w['role'],'output':w['agent_id']} for w in selected}
    outputs=[{'agent_id':w['agent_id'],**info[w['agent_id']],'reviewer_agent_id':ctx['alignment']['reviewer_agent_ids'][w['agent_id']]} for w in selected[::-1]]
    assert node.build_messages({'task':'question'},info,{})==adapter.GSM8KPromptSet.reviewer_messages(task='question',worker_outputs=outputs)


def test_similarity_cache_validates_queries_responses_and_rejects_old_prior(ctx,monkeypatch):
    async def forbidden(*a,**k):pytest.fail('Existing aligned similarity unexpectedly called API')
    matrix=asyncio.run(run.ensure_similarity(ctx))
    assert matrix[0,1]==.6
    item=fake_similarity(ctx);item['pairs'][0]['response']='Score: 8.0'
    with pytest.raises(ValueError,match='audited response'):run._validate_similarity_payload(ctx,item)
    item=fake_similarity(ctx);item['configuration']=dict(item['configuration'],mode='original_prior')
    with pytest.raises(ValueError,match='configuration'):run._validate_similarity_payload(ctx,item)


def fake_result(graph,sample):
    index=int(sample['question'].rsplit(' ',1)[1]);size=graph.num_nodes
    active=graph.spatial_masks.detach().cpu().numpy().reshape(size,size)
    adj=np.triu((torch.rand(size,size).numpy()>.4)*active,1)
    answer=str(index if index%3 else index+1000)
    return {'response':f'The answer is {answer}','spatial_adjacency':adj.tolist(),'events':[],'elapsed_seconds':0.}




def test_atomic_training_resume_matches_uninterrupted_and_retains_orphans(ctx,monkeypatch,tmp_path):
    run.prepare_run(ctx);asyncio.run(run.ensure_similarity(ctx));llm=make_llm()
    batches=0
    async def generate(graph,samples):
        nonlocal batches
        batches+=1
        if batches==3:raise RuntimeError('simulated interruption')
        return [fake_result(graph,s) for s in samples]
    monkeypatch.setattr(run,'run_batch',generate)
    with pytest.raises(RuntimeError,match='interruption'):asyncio.run(run.train(ctx,llm_factory=factory(llm)))
    checkpoint=run._load_checkpoint(ctx['directory']/'training_checkpoint.json',ctx,'training')
    assert len(checkpoint['records'])==8
    resumed=asyncio.run(run.train(ctx,resume=True,llm_factory=factory(llm)))
    other=copy.deepcopy(ctx);other['directory']=tmp_path/'fresh';run.prepare_run(other);asyncio.run(run.ensure_similarity(other))
    direct=asyncio.run(run.train(other,llm_factory=factory(llm)))
    assert resumed['policy']==direct['policy'] and resumed['manager']==direct['manager'] and resumed['rng']==direct['rng']
    assert [r['response'] for r in resumed['records']]==[r['response'] for r in direct['records']]
    assert len(resumed['records'])==40 and resumed['complete']


def test_eval_resumes_full_tail_and_preserves_training_rng(ctx,monkeypatch):
    ctx['alignment']['evaluation_samples']=ctx['alignment']['evaluation_samples'][:7]
    run.prepare_run(ctx);asyncio.run(run.ensure_similarity(ctx));llm=make_llm()
    async def generate(graph,samples):return [fake_result(graph,s) for s in samples]
    monkeypatch.setattr(run,'run_batch',generate)
    training=asyncio.run(run.train(ctx,llm_factory=factory(llm)))
    seen=[]
    async def fail_second(graph,samples):
        seen.append(run.rng_state())
        if len(seen)==2:raise RuntimeError('evaluation interruption')
        return await generate(graph,samples)
    monkeypatch.setattr(run,'run_batch',fail_second)
    with pytest.raises(RuntimeError,match='interruption'):asyncio.run(run.evaluate(ctx,llm_factory=factory(llm)))
    assert seen[0]==training['rng']
    state=run._load_checkpoint(ctx['directory']/'evaluation_checkpoint.json',ctx,'evaluation')
    assert len(state['records'])==4
    monkeypatch.setattr(run,'run_batch',generate)
    summary=asyncio.run(run.evaluate(ctx,resume=True,llm_factory=factory(llm)))
    assert summary['total']==7 and summary['complete']
    state=run._load_checkpoint(ctx['directory']/'evaluation_checkpoint.json',ctx,'evaluation')
    assert [r['sample_index'] for r in state['records']]==list(range(7))
    f=Path(state['records'][0]['trace_json']);f.write_text(f.read_text()+' ')
    with pytest.raises(ValueError,match='modified'):run.summarize(ctx)


def test_changed_cohort_is_rejected(ctx,monkeypatch):
    run.prepare_run(ctx);ctx['args'].resume=True
    new=copy.deepcopy(ctx['alignment']);new['fingerprint']='different-seed0-question-pool'
    monkeypatch.setattr(run,'load_alignment',lambda _:new)
    with pytest.raises(ValueError,match='configuration changed'):run.prepare_context(ctx['args'])


def test_empty_worker_output_is_runtime_failure_as_original_gsm(ctx):
    llm=make_llm()
    llm.tokenizer.batch_decode=lambda tokens,skip_special_tokens:['' for _ in tokens]
    graph=run.build_graph(ctx,llm_factory=factory(llm))
    async def work():
        async with llm.async_batching(4):return await run.run_batch(graph,ctx['alignment']['calibration_samples'][:4])
    with pytest.raises(RuntimeError,match='produced no valid output'):
        asyncio.run(work())
