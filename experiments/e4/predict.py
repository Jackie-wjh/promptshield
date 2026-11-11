"""Small loading interface for M3's app and E6's score recomputation."""
from .common import ROOT, config, score_texts, selected_model


class RoBERTaDetector:
    def __init__(self,device="cpu"):
        self.device=device
        self.model,self.tokenizer,self.selected=selected_model(device)

    def predict_scores(self,texts):
        return score_texts(self.model,self.tokenizer,list(texts),device=self.device)

    def truncation_flags(self,texts):
        ids=self.tokenizer(list(texts),truncation=False,padding=False)["input_ids"]
        return [len(row)>config()["model"]["max_length"] for row in ids]
