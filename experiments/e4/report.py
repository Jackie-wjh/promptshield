"""Assemble result tables and prose from verified measurements, with explicit scope."""
from __future__ import annotations
import re
from .common import (RESULTS,ROOT,config,now,read_json,selected,sha256,source_hashes,write_json)
import pandas as pd


def table(frame,formats=None):
    formats=formats or {}
    lines=["| "+" | ".join(frame.columns)+" |","| "+" | ".join("---" for _ in frame.columns)+" |"]
    for _,row in frame.iterrows():
        values=[]
        for col,v in row.items():
            values.append("—" if pd.isna(v) else (formats[col].format(v) if col in formats else str(v)).replace("|","\\|"))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def main():
    verification=read_json(RESULTS/"verification.json")
    assert verification["status"]=="passed","Verify the saved artifacts before writing the report"
    s,cp=selected();c=config()
    tune=pd.read_csv(RESULTS/"tuning.csv")
    met=pd.read_csv(RESULTS/"metrics_at_0.5.csv").set_index("dataset_split")
    costs=pd.read_csv(RESULTS/"cost_comparison.csv")
    e4gpu=costs[(costs.model_id==s["model_id"])&(costs.device=="cuda")].iloc[0]
    e4cpu=costs[(costs.model_id==s["model_id"])&(costs.device=="cpu")].iloc[0]
    e3gpu=costs[(costs.model_id=="e3_distilbert_base_uncased")&(costs.device=="cuda")].iloc[0]
    d1,d2,d3,d4=[met.loc[n] for n in ["d1_test","d2_test","d3_test","d4_test"]]
    trunc=pd.read_csv(RESULTS/"truncation_stats.csv")
    cases=pd.read_csv(RESULTS/"occlusion_cases.csv")
    spans=pd.read_csv(RESULTS/"occlusion_spans.csv")
    text=(f"RoBERTa's high source-validation performance did not prevent substantial false alarms on external benign text. "
          f"We fine-tuned RoBERTa-base on 11,046 cleaned D1 training examples and compared learning rates of 2e-5 and 5e-5, "
          f"using seed 6405, FP32, batch size 16, a 256-token limit, and three epochs per learning rate. "
          f"The selected checkpoint used learning rate {s['learning_rate']:.0e} at epoch {s['epoch']}, with D1 validation Macro F1 "
          f"of {s['dev_macro_f1']:.4f} at threshold 0.5. Selection was completed before test scoring.\n\n"
          f"At the fixed reference threshold of 0.5, Macro F1 was {d1.macro_f1:.4f} on D1, {d2.macro_f1:.4f} on D2, "
          f"and {d3.macro_f1:.4f} on D3. Attack recall was {d1.attack_recall:.4f}, {d2.attack_recall:.4f}, and {d3.attack_recall:.4f}, "
          f"while false-positive rates were {d1.fpr:.4f}, {d2.fpr:.4f}, and {d3.fpr:.4f}, respectively. "
          f"The benign-only D4 set produced {int(d4.fp)} false positives among 339 examples. These are reference results; "
          "E6 will determine the authoritative validation-selected operating thresholds.\n\n"
          f"On the local RTX 3060 Laptop GPU, end-to-end batch-one median and P95 latency were {e4gpu.median_ms:.2f} and "
          f"{e4gpu.p95_ms:.2f} ms, including tokenization. The supplied DistilBERT checkpoint was remeasured using the same "
          f"device, inputs and precision, giving a median of {e3gpu.median_ms:.2f} ms. "
          f"RoBERTa's saved checkpoint occupied {e4gpu.checkpoint_bytes/1024**2:.1f} MiB; peak training tensor allocation "
          f"was {tune.peak_allocated_mib.max():.0f} MiB.\n\n"
          "Word-span masking on six prespecified E2 case IDs quantified local score sensitivity. Masking can create "
          "out-of-distribution inputs, and these selected cases neither establish causal explanations nor represent all errors. "
          "Only one training seed was used, so training variability remains unmeasured. D2 has only 56 benign and 60 attack "
          "examples, making small differences unstable. Different tokenizers, pretraining and library versions limit architectural "
          "attribution. Scores are uncalibrated, and source-specific annotation policies limit claims about deployment security.")
    words=len(re.findall(r"\b[\w'-]+\b",text))
    assert 200<=words<=400,words
    (RESULTS/"findings_and_limitations.md").write_text("# E4 Findings and Limitations\n\n"+text+"\n",encoding="utf-8")
    selected_metrics=met.loc[["d1_validation","d1_test","d2_test","d3_test"],["n","macro_f1","attack_recall","fpr","tp","fp","tn","fn"]].reset_index()
    for col in ["n","tp","fp","tn","fn"]:selected_metrics[col]=selected_metrics[col].astype(int)
    tuning_table=tune[["learning_rate","epoch","train_loss","dev_loss","dev_macro_f1","dev_attack_recall","dev_fpr","is_selected"]]
    cost_table=costs[["model_id","device","median_ms","p95_ms","peak_allocated_mib","checkpoint_bytes"]].copy()
    cost_table["checkpoint_MiB"]=cost_table.pop("checkpoint_bytes")/1024**2
    top=spans.loc[spans.groupby("sample_id").delta_score.apply(lambda x:x.abs().idxmax())]
    top=top[["sample_id","span","original_score","masked_score","delta_score"]]
    report=["# E4 RoBERTa results and handoff", "",
      "## 中文交接摘要", "",
      f"E4 已完成两个学习率各 3 个 epoch 的完整参数微调。最终选择学习率 **{s['learning_rate']:.0e}、epoch {s['epoch']}**，"
      f"选择依据仅为 D1 验证集在 0.5 阈值下的 Macro F1（{s['dev_macro_f1']:.4f}）。测试集结果未用于选择学习率或 checkpoint。", "",
      "交给 E6 的原始分数共 5,086 行，含 D1 validation、D1/D2/D3 test 和额外的 D4。"
      "本报告使用固定阈值 0.5 的参考结果；E6 的 tF1/t5 尚未提供，因此不能把这里的标签或指标当作全组正式统一结果。", "",
      f"选定模型目录：`{s['checkpoint_dir']}`。权重 SHA-256：`{s['checkpoint_rev']}`。"
      f"自动核验共 {verification['check_count']} 项通过，其中包括重新加载模型复算所有分数。", "",
      "## Methods", "",
      "RoBERTa-base is fine-tuned using the same frozen D1 train/validation IDs as E3. "
      "Both learning-rate conditions use seed 6405, three epochs, max length 256, dynamic padding and FP32 batch 16. "
      "AdamW uses weight decay 0.01 excluding bias/LayerNorm weights, betas (0.9, 0.999), epsilon 1e-8, "
      "10% linear warm-up followed by linear decay, and gradient clipping at 1.0. "
      "Initialization and sample permutations are reset for each condition. All 11,046 training examples are used in each epoch, "
      "including the last incomplete batch (691 optimizer updates per epoch). Six candidates are ranked by validation "
      "Macro F1 at 0.5; ties prefer an earlier epoch, then a smaller learning rate.","",
      f"Pinned model and tokenizer revision: `{s['model_revision']}`. The resource smoke tests used training samples only "
      "and their weights were discarded. Batch 16 was chosen because it fit in local GPU memory, matched E3's physical batch "
      "and was faster than micro-batch 4. E3 used Transformers 5.22.2 / CUDA build cu128; E4 uses Transformers 4.46.3 / cu126. "
      "The two encoders also differ in pretraining, tokenizer and capacity; this is a comparison of practical pipelines.", "",
      "## Learning rate and checkpoint selection", "",table(tuning_table,{k:"{:.5f}" for k in ["train_loss","dev_loss","dev_macro_f1","dev_attack_recall","dev_fpr"]}),"",
      "![Learning curves](figures/fig1_learning_curves.png)","",
      "## Cross source reference results", "",table(selected_metrics,{"macro_f1":"{:.4f}","attack_recall":"{:.4f}","fpr":"{:.4f}"}),"",
      f"**The main transfer finding is the false-alarm increase:** D3 contains {int(d3.fp)} false positives among "
      f"{int(d3.fp+d3.tn)} benign examples (FPR {d3.fpr:.2%}), despite attack recall of {d3.attack_recall:.2%}. "
      f"D3 AUROC is {d3.auroc:.4f}; strong ranking performance therefore does not guarantee a useful fixed operating point. "
      "This observation does not permit selecting a threshold or a different checkpoint on D3. Annotation and input-distribution "
      "differences are plausible contributors, but this experiment does not isolate their causes.","",
      f"D4 benign-only reference result: **FP {int(d4.fp)}/339**, FPR **{d4.fpr:.4f}**. D4 has no attack recall, F1 or AUROC result. "
      "E6 owns final threshold selection and confidence intervals.","",
      "![Reference comparison](figures/fig2_cross_source_reference.png)","",
      "E1/E3 comparison values in this figure are computed from supplied teammate raw-score files. "
      "Only E4's complete scores are independently recomputed by the E4 verification command. Sources are reported separately, "
      "and the figure does not establish statistical superiority or seed stability.","",
      "## Cost and truncation", "",table(cost_table,{k:"{:.2f}" for k in ["median_ms","p95_ms","peak_allocated_mib","checkpoint_MiB"]}),"",
      "Both checkpoints are measured on the same computer, with the same 20 warm-up and 100 measured D1 validation IDs. "
      "Batch-one FP32 timing includes tokenization, transfer, forward pass, softmax and CPU scalar retrieval, "
      "and excludes model loading. CPU timings use four threads. Allocation reflects PyTorch tensors, "
      "not every byte shown by the device driver. The E3 checkpoint hash is checked against its supplied selection record.","",
      f"E4 peak training allocation: {tune.peak_allocated_mib.max():.1f} MiB; peak reserved GPU memory: "
      f"{tune.peak_reserved_mib.max():.1f} MiB. Sum of measured epoch training times: {tune.epoch_train_seconds.sum()/60:.2f} minutes, "
      "excluding validation, checkpoint writing, downloads and setup.","",
      "![Cost comparison](figures/fig3_cost_comparison.png)","",table(trunc,{"tokens_median":"{:.1f}","tokens_p95":"{:.1f}","truncation_rate":"{:.4f}"}),"",
      "## Local word span sensitivity", "",
      "The following six IDs were selected from E2's case list before E4 scoring. They are not assumed to be RoBERTa errors. "
      "Each visible whitespace span is mapped to BPE token positions and replaced by mask tokens without changing sequence length. "
      "The table lists the largest absolute score change for each case. Positive delta indicates lower attack score after masking; "
      "negative delta indicates a higher score. The original and masked scores are independently recomputed during verification.","",
      table(top,{k:"{:.5f}" for k in ["original_score","masked_score","delta_score"]}),"",
      "![Occlusion](figures/fig4_occlusion.png)","",
      "Masking can create unnatural inputs and does not measure interactions between spans. A large change shows local sensitivity "
      "to this intervention; it does not prove a causal explanation, a general keyword shortcut, or the correctness of the dataset label. "
      "The full original texts and E4 predictions are in `occlusion_cases.csv`.","",
      "## Findings and limitations", "",text,"",
      "## Reproduction and division of responsibility", "",
      "See `experiments/e4/README.md` for environment setup and all commands. E6 receives the raw-score handoff and chooses the "
      "final thresholds; E5 consumes `latency.csv`, `model_size.csv` and selected metadata; the app can call "
      "`RoBERTaDetector.predict_scores` and use E6's returned threshold. No E6 threshold result is fabricated.","",
      "This package was developed and executed with AI assistance. Human understanding, contribution records, case review, "
      "AI disclosure and the student's own video explanation remain the student's responsibility. No signatures, instructor "
      "authorization or independent human review are asserted.","",
      "## References", "",
      "- [RoBERTa model card](https://huggingface.co/FacebookAI/roberta-base)",
      "- [Transformers 4.46.3 RoBERTa API](https://huggingface.co/docs/transformers/v4.46.3/model_doc/roberta)",
      "- [PyTorch version installation instructions](https://pytorch.org/get-started/previous-versions/)",
      "- Supplied PromptShield bilingual plan, root README, and E3/E5 handoff packages.",""]
    (RESULTS/"report_section.md").write_text("\n".join(report),encoding="utf-8")
    interpretations={
        "d1:test:000321":"数据集把这条关于行级权限的技术问句标为正常，RoBERTa 也判为正常。遮蔽 How 后分数有所上升，但仍低于 0.5。可描述为对问句形式的局部敏感，不能据此断言模型只依赖 How，或单凭 permissions 等词判定攻击。",
        "d1:test:000441":"数据集标签为正常，模型却以接近 1 的分数判为攻击，属于误报。逐个遮蔽可见词后，分数仍接近 1；没有证据把误报归因于某一个词。接近饱和的 softmax 分数可能压缩变化，且移除单词不一定移除整句的指令含义。这里还体现了缺少角色和授权上下文时的标注边界。",
        "d1:test:000471":"原句被正确判为正常，但遮蔽 What 或 distinguish 会把分数推过 0.5。说明这次遮蔽干预对语句形式和语义联系很敏感。不能据此证明模型理解了完整语境；遮蔽后的文本本身可能不自然。",
        "d1:test:000004":"数据集标签为攻击，但原句分数接近 0，属于漏检。遮蔽 Explain 后分数大幅上升并跨过 0.5。这个变化是局部敏感性的证据，不证明 Explain 在真实输入中具有固定的安全或攻击作用；原句缺少任务上下文，标签含义也必须限定在数据集定义下。",
        "d1:test:000024":"数据集标签为攻击，模型却给出很低的攻击分数。对单个词的遮蔽未能改变分类，因此本组干预没有定位到一个足以改变决定的词。不能把这解释为模型一定使用了语义推理，也不能从该个案确定漏检原因。",
        "d1:test:000086":"原句被正确判为攻击。遮蔽 would、setup 或 display 分别都会大幅降低分数，表明模型对这些位置的干预敏感。单词之间可能存在交互，故不能把三个差值相加当作总贡献，也不能把这些词推广为普遍的攻击触发词。"
    }
    case_doc=["# E4 六个固定案例的解释", "",
      "以下文本和标签来自冻结的 D1 test；分数来自选定 RoBERTa checkpoint。阈值仅使用 0.5 作为参考。"
      "六个 ID 在查看 E4 预测前由 E2 的案例表固定，因此不代表总体错误分布。", ""]
    for case in cases.itertuples():
        g=spans[spans.sample_id==case.sample_id]
        g=g.loc[g.delta_score.abs().nlargest(3).index,["span","original_score","masked_score","delta_score"]]
        pred=int(case.score_attack>=.5)
        result=("TN" if pred==0 else "FP") if case.gold==0 else ("FN" if pred==0 else "TP")
        case_doc.extend([f"## {case.sample_id}","",f"> {case.text}","",
          f"Gold：{case.gold}；预测：{pred}（{result}）；攻击分数：{case.score_attack:.6f}。","",
          table(g,{k:"{:.6f}" for k in ["original_score","masked_score","delta_score"]}),"",
          interpretations[case.sample_id],""])
    (RESULTS/"case_analysis.md").write_text("\n".join(case_doc),encoding="utf-8")
    script=("# E4 真人讲解草稿\n\n请先按你的实际贡献修改措辞，并结合图表用自己的话讲解；该草稿不代替个人理解或 AI 披露。\n\n"
        f"E4 比较了 RoBERTa 的两个学习率，并测量了它的效果和推理成本。我们固定数据、seed、batch 和训练轮数，"
        f"仅用验证集选模型，最终选中学习率 {s['learning_rate']:.0e} 的第 {s['epoch']} 轮。\n\n"
        f"结果表明，较高的同源验证表现不保证跨来源低误报。固定阈值 0.5 下，D3 虽检出全部攻击，"
        f"但误报了 {int(d3.fp)}/{int(d3.fp+d3.tn)} 条正常文本。因此我们同时报告 Recall 和 FPR，"
        "并把最终阈值分析交给 E6，而不根据测试结果反向调参。\n\n"
        f"在同一台机器上，RoBERTa 的 GPU 单条中位延迟为 {e4gpu.median_ms:.2f} 毫秒，"
        f"DistilBERT 为 {e3gpu.median_ms:.2f} 毫秒。六个固定案例的遮蔽图进一步展示了局部敏感性，"
        "但遮蔽会产生不自然文本，所以这不是因果解释。这个实验的价值是明确效果、成本和误报之间的取舍。\n")
    (RESULTS/"presentation_notes_zh.md").write_text(script,encoding="utf-8")
    metadata=read_json(RESULTS/"run_metadata.json")
    metadata.update(status="verified_complete",selected_model=s,source_sha256=source_hashes(),
                    verification={"file":"results/e4/verification.json","check_count":verification["check_count"],"status":"passed"},
                    findings_word_count=words,completed_at_utc=now())
    write_json(RESULTS/"run_metadata.json",metadata)
    write_json(RESULTS/"handoff/run_metadata.json",metadata)
    print(f"Report generated; findings {words} words",flush=True)


if __name__=="__main__":main()
