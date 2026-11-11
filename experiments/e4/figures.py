"""Recreate figures only from saved experimental evidence."""
from __future__ import annotations
from .common import RESULTS, ROOT, metrics, write_csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

COLORS={"E1 Combined":"#707980","E3 DistilBERT":"#258079","E4 RoBERTa":"#3457B1"}


def save(fig,name):
    out=RESULTS/"figures";out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{name}.png",dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","axes.grid":True,
                         "grid.alpha":.18,"axes.axisbelow":True})
    tune=pd.read_csv(RESULTS/"tuning.csv")
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
    for (lr,g),color in zip(tune.groupby("learning_rate"),["#3457B1","#CF6A32"]):
        g=g.sort_values("epoch")
        for ax,col,title in zip(axes,["train_loss","dev_loss","dev_macro_f1"],
                                ["Training loss","Validation loss","Validation Macro F1"]):
            ax.plot(g.epoch,g[col],"o-",label=f"LR {lr:.0e}",color=color)
            ax.set(title=title,xlabel="Epoch",xticks=[1,2,3]);ax.legend(frameon=False)
    selected=tune[tune.is_selected].iloc[0]
    axes[2].scatter([selected.epoch],[selected.dev_macro_f1],marker="*",s=200,color="#202020",zorder=4,label="Selected")
    axes[2].legend(frameon=False)
    axes[2].set_ylim(max(0,tune.dev_macro_f1.min()-.02),1.005)
    fig.suptitle("E4 learning-rate study | D1 validation, fixed threshold 0.5",fontsize=13)
    save(fig,"fig1_learning_curves")
    sources={"E1 Combined":ROOT/"results/e1/handoff/combined_scores.parquet",
             "E3 DistilBERT":ROOT/"results/e3/handoff/distilbert_scores.parquet",
             "E4 RoBERTa":RESULTS/"handoff/roberta_scores.parquet"}
    comparison=[]
    for name,path in sources.items():
        if not path.exists():continue
        raw=pd.read_parquet(path)
        for ds in ["d1","d2","d3"]:
            p=raw[(raw.dataset==ds)&(raw.split=="test")]
            comparison.append({"model":name,"dataset":ds,"source":"supplied_teammate_scores" if name!="E4 RoBERTa" else "this_run",
                               "threshold_policy":"default_0.5_reference_only",**metrics(p.gold,p.score_attack)})
    comp=pd.DataFrame(comparison)
    write_csv(RESULTS/"comparison_at_0.5.csv",comp)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    labels=list(sources);width=.24;x=np.arange(3)
    for j,name in enumerate(labels):
        group=comp[comp.model==name].set_index("dataset").reindex(["d1","d2","d3"])
        for ax,col,title in zip(axes,["macro_f1","attack_recall","fpr"],["Macro F1","Attack recall","False positive rate"]):
            bars=ax.bar(x+(j-1)*width,group[col].to_numpy()*100,width,label=name,color=COLORS[name])
            ax.bar_label(bars,labels=[f"{v:.1f}".removesuffix(".0") for v in group[col].to_numpy()*100],fontsize=8,padding=2)
            ax.set(title=title,xticks=x,xticklabels=["D1","D2","D3"],ylabel="Percent",ylim=(0,112))
    axes[1].legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
    fig.suptitle("Cross-source reference results | fixed threshold 0.5, not E6 final thresholds",fontsize=12)
    save(fig,"fig2_cross_source_reference")
    cost=pd.read_csv(RESULTS/"cost_comparison.csv")
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    model_ids=["e3_distilbert_base_uncased","e4_roberta_base"]
    names=["DistilBERT","RoBERTa"];x=np.arange(2)
    for ax,dev in zip(axes[:2],["cuda","cpu"]):
        g=cost[cost.device==dev].set_index("model_id").reindex(model_ids)
        b1=ax.bar(x-.17,g.median_ms,.34,label="Median",color="#3457B1")
        b2=ax.bar(x+.17,g.p95_ms,.34,label="P95",color="#8CA1D2")
        for bars in [b1,b2]:ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=9)
        ax.set(title=f"{'GPU' if dev=='cuda' else 'CPU'} latency",xticks=x,xticklabels=names,ylabel="Milliseconds")
        ax.margins(y=.22);ax.legend(frameon=False)
    g=cost[cost.device=="cuda"].set_index("model_id").reindex(model_ids)
    bars=axes[2].bar(x,g.peak_allocated_mib,color=["#258079","#3457B1"])
    axes[2].bar_label(bars,fmt="%.0f",padding=3)
    axes[2].set(title="GPU inference peak allocation",xticks=x,xticklabels=names,ylabel="MiB")
    axes[2].margins(y=.22)
    fig.suptitle("Same-device benchmark | batch 1, FP32, tokenization included, 100 trials",fontsize=12)
    save(fig,"fig3_cost_comparison")
    spans=pd.read_csv(RESULTS/"occlusion_spans.csv")
    cases=pd.read_csv(RESULTS/"occlusion_cases.csv")
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout="constrained")
    for ax,case in zip(axes.flat,cases.itertuples()):
        g=spans[spans.sample_id==case.sample_id].copy()
        g=g.loc[g.delta_score.abs().nlargest(8).index].sort_values("delta_score")
        ax.barh(np.arange(len(g)),g.delta_score,color=["#3457B1" if v>=0 else "#CF6A32" for v in g.delta_score])
        ax.set_yticks(np.arange(len(g)),g.span)
        ax.axvline(0,color="#444444",linewidth=.8)
        ax.set_title(f"{case.sample_id} | gold {case.gold}, pred {case.pred_at_0_5 if hasattr(case,'pred_at_0_5') else int(case.score_attack>=.5)}\nAttack score {case.score_attack:.4f}",fontsize=10)
        ax.set_xlabel("Original attack score - masked score")
        ax.grid(axis="y",visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="x",style="sci",scilimits=(-3,3),useMathText=True)
    fig.suptitle("RoBERTa word-span occlusion | fixed E2 case IDs, top 8 absolute changes\nPanel scales differ. Local sensitivity is not a causal explanation.",fontsize=13)
    save(fig,"fig4_occlusion")


if __name__=="__main__":main()
