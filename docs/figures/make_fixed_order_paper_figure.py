#!/usr/bin/env python3
"""Conceptual paper figure: topology, local evidence, topology.

All curves and edge scores are illustrative. Reuses the draft's toy data so
the displayed sparse graph and the displayed importance matrices agree.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from make_fixed_order_failure_method import toy_scores

OUT = Path(__file__).resolve().parent
BLACK, GRAY, LIGHT = "#20262D", "#79818B", "#DCE0E4"
ORANGE, TEAL, RED = "#D98132", "#238674", "#C45B59"
COLORS = ["#507795", "#7E7298", "#55958B", "#B29159"]
FILLS = ["#EDF3F7", "#F2EFF7", "#EFF6F3", "#F8F3EB"]


def text(ax, x, y, s, size=11, color=BLACK, ha="left", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                   fontweight=weight, zorder=20, **kw)


def line(ax, xs, ys, color=LIGHT, lw=1, **kw):
    ax.plot(xs, ys, color=color, lw=lw, **kw)


def arrow(ax, a, b, color=GRAY, lw=1, alpha=1, rad=0, dashed=False, z=3, head=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=head,
                  linewidth=lw, color=color, alpha=alpha, shrinkA=0, shrinkB=0,
                  connectionstyle=f"arc3,rad={rad}",
                  linestyle=(0, (3, 2.5)) if dashed else "-", zorder=z))


def rounded(ax, x, y, w, h, fc="white", ec=LIGHT, lw=.8, radius=.045, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def mark(ax, x, y, *, good, scale=1):
    if good:
        line(ax, [x-.055*scale, x-.01*scale, x+.08*scale],
             [y, y-.045*scale, y+.065*scale], color=TEAL, lw=1.5, zorder=22)
    else:
        d=.048*scale
        line(ax, [x-d, x+d], [y-d, y+d], color=RED, lw=1.4, zorder=22)
        line(ax, [x-d, x+d], [y+d, y-d], color=RED, lw=1.4, zorder=22)


def agent(ax, x, y, i, r=.21, focus=False):
    if focus:
        ax.add_patch(Circle((x,y), r+.09, facecolor="#FFF5E9",
                            edgecolor="#E8BC8D", lw=.8, zorder=5))
    ax.add_patch(Circle((x,y), r, facecolor=FILLS[i], edgecolor=COLORS[i], lw=1.5,zorder=7))
    text(ax,x,y,f"$A_{i+1}$",size=12,ha="center",color=COLORS[i])


def graph(ax, x, sparse=False, masks=None):
    xs = np.linspace(x+.37,x+3.43,4)
    y0,y1 = 5.65,3.75
    tri=np.triu(np.ones((4,4),dtype=bool),1)
    masks=masks if sparse else [tri,tri,np.ones((4,4),dtype=bool)]
    for i in range(4):
        for j in range(4):
            if masks[2][i,j]:
                focus=not sparse and j==3
                arrow(ax,(xs[i],y0-.23),(xs[j],y1+.23),
                      TEAL if sparse else (ORANGE if focus else GRAY),
                      lw=1.15 if sparse or focus else .7,
                      alpha=.76 if sparse else (.55 if focus else .22),
                      dashed=True,head=5)
    for r,y in enumerate([y0,y1]):
        for i in range(4):
            for j in range(i+1,4):
                if masks[r][i,j]:
                    focus=not sparse and r==1 and j==3
                    arrow(ax,(xs[i]+.18,y+.10),(xs[j]-.18,y+.10),
                          TEAL if sparse else (ORANGE if focus else BLACK),
                          lw=1.4 if sparse or focus else .9,
                          alpha=.9 if sparse or focus else .55,
                          rad=-.30-.09*(j-i-1),head=6)
        for i,xx in enumerate(xs):
            agent(ax,xx,y,i,focus=not sparse and r==1 and i==3)
        text(ax,x-.05,y,f"$r={r}$",size=10,color=GRAY,ha="right")
    # Same reviewer connections before and after worker-edge pruning.
    cx=x+1.90
    for xx in xs:
        arrow(ax,(xx,y1-.23),(cx,2.62),color=GRAY,lw=.8,alpha=.45,head=5)
    verts=[(cx,2.69),(cx+.31,2.43),(cx,2.17),(cx-.31,2.43)]
    ax.add_patch(Polygon(verts,facecolor="white",edgecolor=BLACK,lw=1,zorder=6))
    text(ax,cx,2.43,"$R$",size=12,ha="center")
    text(ax,cx,1.93,"reviewer",size=10,color=GRAY,ha="center")
    text(ax,cx,6.70,"$A_1 \\prec A_2 \\prec A_3 \\prec A_4$",size=14,ha="center")
    return xs[-1],y1


def sentence(ax,x,y,w,selected=False):
    rounded(ax,x,y-.095,w,.19,fc="#F3D5B5" if selected else "#F0F1F2",
            ec=ORANGE if selected else "#D9DDE1",lw=.65,radius=.026)
    line(ax,[x+.055,x+w-.055],[y,y],color=ORANGE if selected else "#C3C9CE",lw=1.1,zorder=7)


def local_evidence(ax):
    # Three source messages with visible selected sentences, rather than steps.
    text(ax,5.03,6.46,"Failure attribution",size=13,color=ORANGE,weight="bold")
    text(ax,10.12,5.61,"original output",size=10,color=GRAY,ha="center")
    source_x=5.02
    ys=[5.78,5.13,4.48]
    choices=[{0,2},{1},set()]
    for row,(y,chosen) in enumerate(zip(ys,choices)):
        text(ax,source_x,y,f"$A_{row+1}$",size=12,color=COLORS[row],ha="center")
        for j,w in enumerate([.42,.49,.39]):
            sentence(ax,source_x+.34+j*.54,y,w,selected=j in chosen)
        # Selected-source fractions shown directly as varying edge width.
        arrow(ax,(7.02,y),(8.23,5.13),ORANGE if chosen else "#CCD1D5",
              lw=[2.3,1.4,.7][row],alpha=1 if chosen else .65,
              rad=[-.08,0,.08][row],head=7)
    agent(ax,8.48,5.13,3,r=.25,focus=True)
    arrow(ax,(8.82,5.13),(9.33,5.13),BLACK,lw=1,head=7)
    text(ax,10.12,5.13,r"$\hat y\neq y^\star$",size=15,ha="center")
    mark(ax,10.78,5.14,good=False,scale=1.5)
    text(ax,6.17,4.10,"selected input sentences",size=10,color=ORANGE,ha="center")
    text(ax,8.48,4.61,"receiving worker",size=9.5,color=GRAY,ha="center")

    # Gold-guided evidence search: the one substantial inset in the figure.
    x0,y0,w,h=5.22,2.66,2.08,.91
    line(ax,[x0,x0],[y0,y0+h],color=GRAY,lw=.7)
    line(ax,[x0,x0+w],[y0,y0],color=GRAY,lw=.7)
    ys2=np.array([.06,.25,.57,.86,.64,.74,.41,.31])
    xp=np.linspace(x0,x0+w,len(ys2));yp=y0+ys2*.88
    line(ax,xp,yp,color=ORANGE,lw=1.8)
    ax.scatter(xp,yp,s=10,color=ORANGE,zorder=6)
    ax.scatter([xp[3]],[yp[3]],s=85,marker="*",facecolor=ORANGE,edgecolor="white",lw=.6,zorder=8)
    line(ax,[xp[3],xp[3]],[y0,yp[3]],color=ORANGE,lw=.75,linestyle=(0,(2,2)))
    text(ax,5.22,3.82,r"Gold-prefix score $f(S;y^\star)$",size=10.5)
    text(ax,x0+w/2,2.42,"greedy sentence insertion",size=9.5,color=GRAY,ha="center")
    text(ax,xp[3]+.18,yp[3]+.15,"$S^*$",size=12,color=ORANGE)

    # Validation is an experimental check on selected evidence, not another box.
    arrow(ax,(7.49,3.15),(8.03,3.15),ORANGE,lw=1.1,head=7)
    text(ax,9.48,3.82,"Gold-free verification",size=11,ha="center")
    text(ax,8.22,3.17,"$S^*$",size=13,color=ORANGE)
    arrow(ax,(8.53,3.15),(8.91,3.15),GRAY,lw=.8,head=6)
    agent(ax,9.19,3.15,3,r=.18)
    mark(ax,9.88,3.38,good=True,scale=1.1)
    text(ax,10.10,3.38,"token ratio",size=10.5,color=TEAL)
    mark(ax,9.88,2.93,good=False,scale=1.1)
    text(ax,10.10,2.93,"observed 0",size=10.5,color=RED)
    text(ax,9.48,2.42,"original prompt; no gold input",size=9.5,color=GRAY,ha="center")

    # Small correct-node reference; no second branch or routing decision.
    line(ax,[4.91,11.18],[2.06,2.06],color=LIGHT,lw=.7)
    text(ax,5.03,1.76,"Correct nodes",size=10.5,color=TEAL,weight="bold")
    mark(ax,6.53,1.76,good=True)
    text(ax,6.77,1.76,"recover the recorded-output score",size=10.2,color=GRAY)
    text(ax,11.16,1.76,"$\\rho \\geq 0.95$",size=11,color=TEAL,ha="right")


def matrix(ax,a,valid,x,y,label,cell=.145):
    cmap=LinearSegmentedColormap.from_list("edge",["#FFFFFF","#BBDDD4",TEAL,"#24574C"])
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((x+j*cell,y+(3-i)*cell),cell,cell,
                        facecolor=cmap(float(a[i,j])) if valid[i,j] else "#E5E7EA",
                        edgecolor="white",lw=.5,zorder=5,
                        hatch=None if valid[i,j] else "//"))
    text(ax,x+2*cell,y+.78,label,size=11,ha="center")


def render():
    plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"stix",
                         "svg.fonttype":"none","pdf.fonttype":42})
    arrays,valid,edges,masks=toy_scores()
    fig=plt.figure(figsize=(16,7.9),facecolor="white")
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,7.9),aspect="equal");ax.axis("off")
    text(ax,.48,7.47,"(a)",size=14,weight="bold")
    text(ax,1.02,7.47,"Dense communication",size=15,weight="bold")
    text(ax,4.91,7.47,"(b)",size=14,weight="bold")
    text(ax,5.45,7.47,"Local attribution evidence",size=15,weight="bold")
    text(ax,12.08,7.47,"(c)",size=14,weight="bold")
    text(ax,12.62,7.47,"Sparse communication",size=15,weight="bold")
    focus=graph(ax,.65)
    graph(ax,12.05,sparse=True,masks=masks)

    # Fine magnification guides, not inter-stage workflow arrows.
    line(ax,[focus[0]+.31,4.65],[focus[1]+.16,6.25],color="#D3B38F",lw=.85,linestyle=(0,(3,3)),zorder=1)
    line(ax,[focus[0]+.31,4.65],[focus[1]-.16,2.26],color="#D3B38F",lw=.85,linestyle=(0,(3,3)),zorder=1)
    local_evidence(ax)

    # Compact quantitative spine below the conceptual objects.
    line(ax,[.48,15.65],[1.32,1.32],color=LIGHT,lw=.8)
    text(ax,.59,.94,r"$a_e(x)=|T_e\cap T(S^*)|\,/\,|T_e|$",size=14)
    text(ax,.59,.55,"correct / rescued; otherwise observed zero",size=9.2,color=GRAY)
    text(ax,4.75,.80,r"$\bar a_e=\mathrm{mean}_{x:\,e\ \mathrm{observed}}\,a_e(x)$",size=14,ha="center")
    text(ax,4.75,.38,"all calibration samples",size=9.5,color=GRAY,ha="center")
    for a,v,x,label in zip(arrays,valid,[6.90,7.75,8.60],
                          [r"$\bar M_s^{(0)}$",r"$\bar M_s^{(1)}$",r"$\bar M_t^{(1)}$"]):
        matrix(ax,a,v,x,.38,label)
    arrow(ax,(9.46,.73),(10.10,.73),GRAY,lw=1,head=7)
    text(ax,11.22,.94,"Global bottom-p%",size=11,ha="center")
    text(ax,11.22,.55,"whole-edge pruning",size=9.5,color=GRAY,ha="center")
    # Link the statistical mask to the sparse graph without a pipeline panel.
    arrow(ax,(12.30,.74),(13.65,1.63),TEAL,lw=1,rad=.15,head=7)
    text(ax,15.60,.93,"Fixed order",size=10.5,ha="right")
    text(ax,15.60,.55,"Frozen mask · dense reviewer",size=9.5,color=GRAY,ha="right")
    text(ax,15.60,.15,"Schematic; illustrative scores and curves",size=8.5,color="#979DA4",ha="right")
    for ext in ["svg","pdf","png"]:
        path=OUT/f"fixed_order_paper_v2.{ext}"
        fig.savefig(path,facecolor="white",dpi=200)
        print(path)
    plt.close(fig)


if __name__=="__main__":
    render()
