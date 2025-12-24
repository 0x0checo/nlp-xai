import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

class IGExplainer:
    def __init__(self, ckpt_dir='ckpt_sst2_bert', max_length=128, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, output_attentions=False
        ).to(self.device)
        self.model.eval()

        # For LayerIntegratedGradients, we use the embeddings layer
        self.emb_layer = self.model.bert.embeddings

    def _forward(self, input_ids, attention_mask, target_label):
        """
        :param input_ids:
        :param attention_mask:
        :param target_label:
        :return: logits for a specific class
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        return logits[:, target_label]

    @torch.no_grad()
    def _predict(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probs = F.softmax(outputs.logits, dim=-1)
        # Get the class index with the highest probability
        pred_label = int(probs.argmax(dim=-1)[0].item())
        # Get the specific probability value (confidence) for that predicted class
        pred_prob = float(probs[0, pred_label].item())
        return pred_label, pred_prob

    def explain(self, text, topk=10, n_steps=50):
        """
        Compute ig for one input text
        """
        # Encode text
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            # Fixed length is convenient for baselines
            padding='max_length',
            truncation=True,
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        # Get prediction
        with torch.no_grad():
            pred_label, pred_prob = self._predict(input_ids, attention_mask)
        # Prepare baseline
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.unk_token_id
        # Create a baseline tensor of the same shape as input_ids
        baseline_ids = torch.full_like(input_ids, pad_id).to(self.device)

        # Define the forward function for IG and return logits
        lig = LayerIntegratedGradients(
            lambda ids, mask: self._forward(ids, mask, target_label=pred_label),
            self.emb_layer,
        )

        # Compute attributions
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True
        )
        attributions = attributions[0]

        # Aggregate over embedding dimension
        token_attrib = attributions.sum(dim=-1)
        token_attrib = token_attrib.abs()

        # Map to tokens & filter specials
        input_ids_list = input_ids[0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids_list)
        specials = {
            tok for tok in
            [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]
            if tok is not None
        }

        pairs = []
        for token, score, mask in zip(tokens, token_attrib.tolist(),
                                      attention_mask[0].cpu().tolist()):
            # Skip padding positions and special tokens
            if mask == 0:
                continue
            if token in specials:
                continue
            pairs.append((token, float(score)))

        if not pairs:
            df = pd.DataFrame(columns=['token', 'ig_score'])
            return df, pred_label, pred_prob

        # Normalize scores
        total = sum(s for _, s in pairs) or 1.0
        pairs_norm = [(t, s / total) for t, s in pairs]

        # Sort top-k pairs
        pairs_norm.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs_norm[:topk]

        df = pd.DataFrame(top_pairs, columns=['token', 'ig_score'])
        return df, pred_label, pred_prob


if __name__ == "__main__":
    ckpt = "/proj/uppmax2020-2-2/wenyuli/outputs/ckpt_sst2_bert/checkpoint-6315"
    explainer = IGExplainer(ckpt_dir=ckpt, max_length=128)

    text = "It's a charming and often affecting journey."
    df, yhat, p = explainer.explain(text, topk=10, n_steps=50)

    print("Text:", text)
    print("Predicted label:", yhat, "prob:", f"{p:.4f}")
    print(df)
