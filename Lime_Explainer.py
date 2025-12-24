import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

class LimeSST2Explainer:
    def __init__(self, ckpt_dir, max_length=128, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        # Load tokenizer and model from my fine_tuned checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(self.device)
        self.model.eval()

        # For SST-2, class 0 = negative, 1 = positive
        self.class_names = ['negative', 'positive']
        # Create lime explainer
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_proba(self, texts):
        """
        LIME requires a function that takes a list[str] and returns
        an array shape (n_samples, n_classes) with probabilities.
        """
        # If LIME calls with a single string
        if isinstance(texts, str):
            texts = [texts]
        # Encode texts
        enc = self.tokenizer(
            list[texts],
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        # Get probabilities
        with torch.no_grad:
            outputs = self.model(**enc)
            probs = F.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    def explain(self, text, topk=10, num_samples=500):
        """
                Generate LIME explanation for a single text.
        """
        # Get predictions
        probs = self.predict_proba([text])
        pred_label = int(np.argmax(probs[0]))
        pred_prob = float(probs[0, pred_label])

        # Get explanations
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_proba,
            labels=(pred_label,),
            num_features=topk,
            num_samples=num_samples,
        )

        # Extract attribution weights with tokens
        try:
            feat_list = exp.as_list(label=pred_label)
        except TypeError:
            feat_list = exp.as_list()

        if not feat_list:
            df = pd.DataFrame(columns=['tokens', 'lime_score'])
            return df, pred_label, pred_prob

        tokens, scores = zip(*feat_list)
        df = pd.DataFrame({'token': list(tokens), 'lime_score': list(scores)})
        return df, pred_label, pred_prob
