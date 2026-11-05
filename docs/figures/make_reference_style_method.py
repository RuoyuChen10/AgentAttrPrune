#!/usr/bin/env python3
"""Reference-style method composite built from actual fixed-order MMLU artifacts.

Shortened dialogue and sentence excerpts are explicitly marked. Matrix values,
failure-attribution curve, source-token ratios and pruning masks are real.
No inference, attribution or evaluation jobs are launched by this script.
"""
from pathlib import Path
import hashlib
import json
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware'
BLACK,GRAY,LIGHT='#17202C','#65717F','#DFE4EA'
ORANGE,ORANGE_PALE,TEAL,RED,BLUE='#ED812E','#FFE0BC','#21856F','#C95755','#4B8BC9'
ROLE_COLORS=['#2888CE','#ED8A28','#8457C5','#27A6A8','#718349']
IDS=['knowledgeable_expert','critic','mathematician','psychologist','historian']
NAMES=['Knowledge Expert','Critic','Mathematician','Psychologist','Historian']


def load(path):
    return json.loads(Path(path).read_text())


def txt(ax,x,y,s,size=10,color=BLACK,ha='left',weight='normal',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight,zorder=30,**kw)


def line(ax,xs,ys,color=LIGHT,lw=1,**kw):
    ax.plot(xs,ys,color=color,lw=lw,**kw)


def rect(ax,x,y,w,h,fc='white',ec=LIGHT,lw=.8,r=.045,z=4):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax,a,b,color=GRAY,lw=1,rad=0,alpha=1,head=7,dashed=False,z=3):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',color=color,linewidth=lw,
                  mutation_scale=head,connectionstyle=f'arc3,rad={rad}',
                  shrinkA=0,shrinkB=0,alpha=alpha,
                  linestyle=(0,(3,2)) if dashed else '-',zorder=z))


def broad_arrow(ax,x,y,dx=.36,dy=0):
    ax.add_patch(FancyArrowPatch((x,y),(x+dx,y+dy),arrowstyle='simple',
                  mutation_scale=25,facecolor='#92BCE4',edgecolor='#4B86BE',
                  linewidth=.8,zorder=12))


def robot(ax,x,y,i,scale=1):
    c=ROLE_COLORS[i];s=scale
    line(ax,[x,x],[y+.16*s,y+.27*s],color=c,lw=1.1,zorder=10)
    ax.add_patch(Circle((x,y+.30*s),.034*s,facecolor='white',edgecolor=c,lw=1,zorder=12))
    rect(ax,x-.19*s,y-.025*s,.38*s,.235*s,fc=c,ec=c,r=.055*s,z=12)
    rect(ax,x-.145*s,y+.015*s,.29*s,.14*s,fc='#1B2E45',ec='white',r=.032*s,lw=.55,z=13)
    for d in [-.065,.065]:
        ax.add_patch(Circle((x+d*s,y+.09*s),.022*s,facecolor='white',edgecolor='none',zorder=15))
    rect(ax,x-.125*s,y-.17*s,.25*s,.115*s,fc=c,ec=c,r=.025*s,z=12)
    ax.add_patch(Circle((x,y-.11*s),.027*s,facecolor='white',edgecolor='none',zorder=15))


def heading(ax,x,y,letter,title):
    txt(ax,x,y,f'({letter})',size=14,weight='bold')
    txt(ax,x+.47,y,title,size=13.5,weight='bold')


def snippet(s,n=92):
    s=' '.join(s.split()).replace('Reasoning: ','')
    return textwrap.shorten(s,width=n,placeholder=' …')


def decision(s):
    m=re.search(r'Decision:\s*([A-D])',s)
    return m.group(1) if m else '?'


def matrix(ax,data,x,y,side=1.43,title='',valid=None,mask=None,numbers=False,
           labels=True,vmax=.7,binary=False,tiny=False):
    n=5;cell=side/n
    cmap=plt.get_cmap('viridis');norm=colors.Normalize(vmin=0,vmax=1 if binary else vmax)
    valid=np.ones((n,n),dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    for i in range(n):
        for j in range(n):
            v=float(data[i,j]);exists=bool(valid[i,j]);kept=mask is None or bool(mask[i,j])
            fc=cmap(norm(v)) if exists and kept else ('#F8FAFC' if exists else '#E4E9EF')
            xx=x+j*cell;yy=y+(n-i-1)*cell
            ax.add_patch(Rectangle((xx,yy),cell,cell,facecolor=fc,edgecolor='white',linewidth=.4,zorder=4))
            if exists and not kept:
                d=cell*.11;cx=xx+cell/2;cy=yy+cell/2
                line(ax,[cx-d,cx+d],[cy-d,cy+d],color=RED,lw=.6,zorder=6)
                line(ax,[cx-d,cx+d],[cy+d,cy-d],color=RED,lw=.6,zorder=6)
            elif exists and numbers:
                text=str(int(v)) if binary else f'{v:.2f}'
                txt(ax,xx+cell/2,yy+cell/2,text,size=6.3 if not tiny else 4.8,
                    color='white' if norm(v)<.64 else BLACK,ha='center')
    ax.add_patch(Rectangle((x,y),side,side,fill=False,edgecolor='#8592A1',lw=.65,zorder=7))
    txt(ax,x+side/2,y+side+.20,title,size=9 if not tiny else 7.2,ha='center')
    if labels:
        for i in range(n):
            txt(ax,x-.09,y+(n-i-.5)*cell,f'$A_{i+1}$',size=6.9 if not tiny else 5.6,ha='right',color=GRAY)
            txt(ax,x+(i+.5)*cell,y-.15,f'$A_{i+1}$',size=6.9 if not tiny else 5.6,ha='center',color=GRAY)


def matrix_group(ax,spatial,temporal,x,y,side=1.43,gap=.48,row_gap=.62,
                 masks=None,numbers=False,tiny=False,vmax=.7):
    for r in [0,1]:
        yy=y+(1-r)*(side+row_gap)
        matrix(ax,spatial[r],x,yy,side,title=f'r={r} · spatial',
               valid=np.triu(np.ones((5,5),bool),1),numbers=numbers,tiny=tiny,vmax=vmax,
               mask=None if masks is None else masks['spatial'][r])
        matrix(ax,temporal[r],x+side+gap,yy,side,title=f'r={r} · temporal',
               valid=np.full((5,5),r>0),numbers=numbers,tiny=tiny,vmax=vmax,
               mask=None if masks is None else masks['temporal'][r])


def communication(ax,example,trace):
    heading(ax,.30,10.00,'a','Fixed-order multi-agent system')
    txt(ax,2.47,9.58,r'$A_1\prec A_2\prec A_3\prec A_4\prec A_5$',size=13,ha='center')
    coords=[(.92,8.30),(2.47,8.30),(4.02,8.30),(1.65,6.67),(3.27,6.67)]
    for i in range(5):
        for j in range(i+1,5):
            a,b=coords[i],coords[j]
            if abs(a[1]-b[1])<.1:
                arrow(ax,(a[0]+.19,a[1]+.15),(b[0]-.22,b[1]+.15),
                      color='#7E8D9D',lw=.85,rad=-.22-.08*(j-i-1),alpha=.8,head=5)
            else:
                arrow(ax,(a[0],a[1]-.2),(b[0],b[1]+.30),color='#9CA8B5',lw=.7,alpha=.55,head=5)
    ev={e['agent_id']:e for e in trace['events'] if e.get('round')==1 and e.get('agent_id') in IDS}
    for i,(x,y) in enumerate(coords):
        robot(ax,x,y,i,scale=1.05)
        txt(ax,x,y-.34,f'$A_{i+1}$  {NAMES[i]}',size=7.5,color=ROLE_COLORS[i],ha='center',weight='bold')
        out=ev[IDS[i]].get('output_text',ev[IDS[i]].get('output',''))
        if isinstance(out,dict):out=out.get('text','')
        if not out:
            out=next((m['source_output'] for m in example['incoming_messages']
                      if m['agent_id']==IDS[i] and m['round']==1),example['original']['text'])
        by=8.85 if i<3 else 5.45
        rect(ax,x-.69,by,1.38,.64,ec=ROLE_COLORS[i],lw=.8)
        reasoning=next((part for part in re.split(r'(?<=[.!?])\s+',out)
                        if 'Option ' in part),out)
        body='\n'.join(textwrap.wrap(snippet(reasoning,49),width=28))
        txt(ax,x-.60,by+.42,body,size=6.7,linespacing=1.35)
        txt(ax,x-.60,by+.12,f'Decision: {decision(out)}',size=7.1,weight='bold')
        if i<3:line(ax,[x,x],[by,y+.33],color=ROLE_COLORS[i],lw=.7)
        else:line(ax,[x,x],[by+.64,y-.45],color=ROLE_COLORS[i],lw=.7)
    txt(ax,2.47,5.10,'Previous-round message history',size=10,ha='center',weight='bold')
    rect(ax,.38,4.21,4.18,.66,ec='#7B8794',fc='#F8FAFC')
    history={m['agent_id']:m for m in example['incoming_messages'] if m['round']==0}
    for i in range(5):
        x=.47+i*.82
        rect(ax,x,4.29,.73,.49,ec=ROLE_COLORS[i],lw=.65,fc='white',r=.025)
        txt(ax,x+.365,4.66,f'$A_{i+1}$',size=9,ha='center',color=ROLE_COLORS[i])
        txt(ax,x+.365,4.43,f"Decision: {decision(history[IDS[i]]['source_output'])}",size=6.9,ha='center')
    txt(ax,2.47,3.67,'Dense candidate communication',size=10,ha='center',weight='bold')
    spatial=np.triu(np.ones((5,5)),1);temporal=np.ones((5,5))
    # Binary zero cells are valid zeros here, matching the user's reference.
    matrix(ax,spatial,.66,1.63,side=1.62,title='Spatial',numbers=True,binary=True)
    matrix(ax,temporal,2.69,1.63,side=1.62,title='Temporal (r > 0)',numbers=True,binary=True)
    txt(ax,2.48,1.18,'Rows: source   ·   Columns: target',size=8.2,color=GRAY,ha='center')
    txt(ax,2.48,.72,'All original messages are preserved.\nReviewer edges remain outside the pruning budget.',size=8.4,color=GRAY,ha='center',linespacing=1.6)


def attribution(ax,example):
    heading(ax,5.23,10.00,'b','Correctness-aware attribution')
    txt(ax,5.27,9.62,'MMLU #25 · Psychologist · round 1',size=9.5,color=GRAY)
    txt(ax,10.61,9.62,'Source-token ratio (%)',size=8,color=GRAY,ha='right')
    rect(ax,5.27,8.67,.28,.73,fc='#E4E7EA',ec='#86909B',r=.012)
    txt(ax,5.41,9.035,'TASK',size=8,ha='center',rotation=90,weight='bold')
    rect(ax,5.61,8.67,5.00,.73,ec='#86909B',r=.018)
    task='\n'.join(textwrap.wrap(snippet(example['sample']['question'],210),width=90))
    txt(ax,5.74,9.10,task,size=7.4,linespacing=1.45)
    txt(ax,5.74,8.79,'Task/prompt fixed · excerpts shown · ratios: selected / visible source tokens',size=7.1,color=GRAY)
    by_index={s['index']:s for s in example['sentences']}
    selected_indices=example['report_excerpt_indices']
    cursor=8.52
    group_bounds={}
    for k,edge in enumerate(example['edge_scores']):
        idx=IDS.index(edge['source_agent'])
        group=edge['edge_type']
        entries=[by_index[i] for i in selected_indices if
                 by_index[i]['source_agent']==edge['source_agent'] and
                 by_index[i]['source_round']==edge['source_round']]
        if not entries:
            entries=[next(s for s in example['sentences'] if s['incoming_index']==k)]
        height=.31+.175*len(entries)
        bottom=cursor-height
        group_bounds.setdefault(group,[bottom,cursor]);group_bounds[group][0]=bottom
        robot(ax,5.82,bottom+height/2-.04,idx,scale=.58)
        txt(ax,6.17,cursor-.13,NAMES[idx],size=7.5,color=ROLE_COLORS[idx],weight='bold')
        txt(ax,10.50,cursor-.13,f"{edge['token_ratio']*100:.1f}%",size=8,ha='right',weight='bold')
        rect(ax,6.11,bottom,4.50,height,ec=ROLE_COLORS[idx],lw=.65,r=.018)
        for q,s in enumerate(entries):
            yy=cursor-.32-q*.175
            rect(ax,6.20,yy-.073,4.31,.145,fc=ORANGE_PALE if s['selected'] else '#F2F3F5',
                 ec=ORANGE if s['selected'] else 'none',lw=.45,r=.012)
            txt(ax,6.25,yy,snippet(s['text'],86),size=6.7,color=BLACK if s['selected'] else '#7E8792')
        cursor=bottom-.075
        if k==2:cursor-=.06
    for kind,(bottom,top) in group_bounds.items():
        rect(ax,5.27,bottom,.28,top-bottom,fc='#E4E7EA',ec='#86909B',r=.01)
        txt(ax,5.41,(top+bottom)/2,'CURRENT ROUND' if kind=='spatial' else 'HISTORY',size=7.8,
            rotation=90,ha='center',weight='bold')
    output_y=cursor-.10
    txt(ax,5.33,output_y,'Original receiver: D',size=9.4,color=RED,weight='bold')
    txt(ax,10.54,output_y,'Gold: C',size=9.4,color=TEAL,ha='right',weight='bold')

    # Real complete insertion curve. No probability or illustrative surrogate.
    xx=5.63;yy=1.80;ww=2.80;hh=1.00
    curve=example['curve'];steps=np.asarray(curve['steps']);values=np.asarray(curve['insertion'])
    lo=float(values.min())-2;hi=float(values.max())+2
    px=xx+steps/steps.max()*ww;py=yy+(values-lo)/(hi-lo)*hh
    line(ax,[xx,xx],[yy,yy+hh],color=BLACK,lw=.7)
    line(ax,[xx,xx+ww],[yy,yy],color=BLACK,lw=.7)
    line(ax,px,py,color=BLACK,lw=1.1)
    peak=example['peak']['step'];peakx=px[peak];peaky=py[peak]
    line(ax,[peakx,peakx],[yy,peaky],color=ORANGE,lw=1,linestyle=(0,(2,2)))
    ax.scatter([peakx],[peaky],color=ORANGE,s=33,zorder=15)
    for step in [0,20,42]:
        txt(ax,xx+step/42*ww,yy-.15,str(step),size=7,ha='center',color=GRAY)
    for value in [int(np.ceil(lo/5)*5),int(np.floor(hi/5)*5)]:
        txt(ax,xx-.08,yy+(value-lo)/(hi-lo)*hh,str(value),size=7,ha='right',color=GRAY)
    txt(ax,xx,yy+hh+.23,'Failure: gold-prefix logit score',size=9,weight='bold')
    txt(ax,xx+ww/2,yy-.39,'Inserted sentences',size=8,color=GRAY,ha='center')
    txt(ax,peakx+.13,peaky-.16,'peak: 6 / 42',size=7.5,color=ORANGE)

    robot(ax,9.03,2.55,3,scale=.82)
    txt(ax,9.48,2.95,'Gold-free replay',size=9,weight='bold')
    txt(ax,9.48,2.62,'Decision: C',size=11,color=TEAL,weight='bold')
    txt(ax,9.48,2.29,'Original prompt',size=8,color=GRAY)
    arrow(ax,(8.57,2.52),(8.77,2.52),color=ORANGE,lw=1.1)
    txt(ax,9.66,1.73,'Correct nodes use\nrecorded-output recovery\n($\\rho\\geq0.95$).',size=7.6,color=GRAY,ha='center',linespacing=1.35)
    line(ax,[5.27,10.61],[1.22,1.22],color=LIGHT,lw=.8)
    txt(ax,5.33,.94,'Rescued → source-token ratios',size=9,color=TEAL,weight='bold')
    txt(ax,5.33,.59,'Rescue failed → observed zeros',size=9,color=RED,weight='bold')
    txt(ax,10.59,.22,'Local replay does not overwrite the dense trace.',size=7.8,color=GRAY,ha='right')


def per_sample(ax,samples):
    heading(ax,11.25,10.00,'c','Sample matrices')
    for sample,y in zip(samples,[6.75,3.85,.95]):
        txt(ax,12.61,y+2.62,f"Sample {sample['sample_index']:02d}",size=9.2,ha='center',weight='bold')
        m=sample['matrices']['token_ratio']
        matrix_group(ax,np.array(m['spatial']),np.array(m['temporal']),11.63,y,
                     side=.83,gap=.42,row_gap=.55,tiny=True,vmax=1.0)
    txt(ax,12.62,.46,'… all 40 calibration samples',size=8.5,color=GRAY,ha='center')
    txt(ax,12.62,.19,'Sample color scale: 0–1',size=7.4,color=GRAY,ha='center')


def aggregated(ax,aggregate,policy):
    heading(ax,14.82,10.00,'d','Aggregate and prune')
    m=aggregate['metrics']['token_ratio'];s=np.array(m['spatial']['conditional_mean']);t=np.array(m['temporal']['conditional_mean'])
    txt(ax,17.10,9.64,'Mean importance over observed edges',size=10,ha='center',weight='bold')
    matrix_group(ax,s,t,15.19,6.12,side=1.25,gap=.58,row_gap=.50,numbers=True)
    gradient=np.linspace(0,.7,120).reshape(-1,1)
    ax.imshow(gradient,extent=(18.94,19.06,6.12,9.12),origin='lower',aspect='auto',cmap='viridis',vmin=0,vmax=.7,zorder=4)
    for v in [0,.2,.4,.7]:txt(ax,19.15,6.12+v/.7*3.00,f'{v:.1f}',size=7,color=GRAY)
    txt(ax,19.55,7.80,'Token-ratio importance',size=8,ha='center',rotation=90)
    broad_arrow(ax,17.30,5.91,dx=0,dy=-.43)
    txt(ax,17.10,5.18,'Global 50% pruning · 22 / 45 edges removed',size=9.3,ha='center',weight='bold')
    txt(ax,17.10,4.84,'One ranking across rounds and edge types',size=8.1,color=GRAY,ha='center')
    masks={k:np.array(v) for k,v in policy['masks'].items()}
    matrix_group(ax,s,t,15.19,1.23,side=1.25,gap=.58,row_gap=.50,numbers=True,masks=masks)
    txt(ax,17.10,.79,'× pruned     colored: retained     gray: absent',size=8.1,color=GRAY,ha='center')
    txt(ax,17.10,.40,'Frozen policy · fixed execution order',size=9.3,ha='center',weight='bold')


def render():
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix',
                         'svg.fonttype':'none','pdf.fonttype':42})
    example=load(HERE/'failure_attribution_example_data.json')
    trace=load(example['provenance']['trace_json'])
    aggregate_path=RUN/'attribution/aggregate_edge_matrices.json'
    policy_path=RUN/'evaluation/static_token_ratio_conditional_mean_global_prune_50pct_val/resolved_policy.json'
    aggregate=load(aggregate_path);policy=load(policy_path)
    assert aggregate['aggregation_scope']=='all' and aggregate['num_aggregated_samples']==40
    assert policy['pruning_allocation']=='global'
    assert hashlib.sha256(aggregate_path.read_bytes()).hexdigest()==policy['static_sweep']['aggregate_sha256']
    assert example['verification']['correct'] and not example['verification']['gold_supplied_to_generation']
    sample_paths=[RUN/f'attribution/sample_{i:05d}/sample_edge_matrices.json' for i in [25,11,17]]
    samples=[load(p) for p in sample_paths]
    fig=plt.figure(figsize=(20,10.5),facecolor='white');ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,20),ylim=(0,10.5));ax.axis('off')
    communication(ax,example,trace)
    attribution(ax,example)
    per_sample(ax,samples)
    aggregated(ax,aggregate,policy)
    broad_arrow(ax,4.73,7.32,dx=.35)
    broad_arrow(ax,10.78,7.32,dx=.35)
    broad_arrow(ax,14.27,7.32,dx=.36)
    # Sparse dividers preserve the reference's montage hierarchy.
    for x in [4.96,10.97,14.49]:line(ax,[x,x],[.42,9.49],color='#EDF0F3',lw=.7,zorder=0)
    stem=HERE/'fixed_order_reference_v3'
    outputs=[]
    for ext in ['svg','pdf','png']:
        path=stem.with_suffix('.'+ext);fig.savefig(path,facecolor='white',dpi=220);outputs.append(str(path));print(path)
    plt.close(fig)
    provenance={'figure_outputs':outputs,'example':str(HERE/'failure_attribution_example_data.json'),
                'aggregate':str(aggregate_path),'aggregate_sha256':hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
                'policy':str(policy_path),'sample_matrices':[str(p) for p in sample_paths],
                'notes':['All matrices, scores, curve points and masks use existing artifacts.',
                         'Dialogues and displayed source sentences are shortened excerpts.',
                         'Binary source graph masks are exact structural definitions.']}
    stem.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':render()
