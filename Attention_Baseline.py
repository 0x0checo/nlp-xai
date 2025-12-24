import argparse, csv, os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create a backup special token set
SPECIAL_FALLBACK = {'[CLS]', '[SEP]', '[PAD]'}

# Define a function to calculate attention scores from [CLS] to other tokens
# Disables gradient calculation for this function
@torch.no_grad()
def cls_attn_scores(outputs, attention_mask):
    last = outputs.attentions[-1] # Get the last layer attention tensor
    att = last.mean(dim=1)[:,0,:] # Head average and take [CLS] row
    att = att * attention_mask # Let padding mask = 0 of attention
    denom = att.sum(dim=-1, keepdim=True).clamp(min=1e-12) # Sum row scores, keep dim, clamp to avoid division by zero
    return att/denom # Return the normalized attention scores

# Define a function to convert a row of token IDs back to token strings
def tokens_from_ids(tokenizer, input_ids_row):
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist()) # # Convert the PyTorch tensor row to a Python list, then convert IDs to token strings
    # Filter special tokens
    specials = {
        token for token in
        [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
        if token is not None
    }
    if not specials:
        specials = SPECIAL_FALLBACK
    return tokens, specials

# Define a function to rank top-k token-score pairs
def topk_pairs(tokens, scores, specials, k):
    pairs = [(t, float(s)) for t, s in zip(tokens, scores) if t not in specials]
    pairs.sort(key=lambda x: x[1], reverse=True) # Using score to rank in descending order
    return pairs[:k]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default='/proj/uppmax2020-2-2/wenyuli/outputs/ckpt_sst2_bert/checkpoint-6315')
    ap.add_argument("--split", default="validation", choices=["validation", "test", "train"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_csv", default="/proj/uppmax2020-2-2/wenyuli/outputs/explainability_sst2/sst2_val_attn.csv")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir,use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt_dir,output_attentions=True).to(device).eval()

    ds = load_dataset('glue', 'sst2')[args.split]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    # Create a csv file and add row of names
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "text", "pred_label", "pred_prob",
                "topk_tokens", "topk_weights"
            ]
        )
        writer.writeheader()

        # Loop through the dataset in batches
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start: start + args.batch_size]
            texts = batch['sentence']

            enc = tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=args.max_length,
                padding=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_labels = probs.argmax(dim=-1).tolist()
            pred_probs = probs.max(dim=-1).values.tolist()

            attention = cls_attn_scores(outputs, enc['attention_mask']) # Get [CLS] attention scores

            for i, text in enumerate(texts):
                # Get the list of token strings and special tokens
                tokens, specials = tokens_from_ids(tokenizer, enc['input_ids'][i])
                top_pairs = topk_pairs(tokens, attention[i].tolist(), specials, args.topk)
                writer.writerow({
                    "id": start + i,
                    "text": text,
                    "pred_label": pred_labels[i],
                    "pred_prob": f"{pred_probs[i]:.6f}",
                    "topk_tokens": ";".join(t for t, _ in top_pairs),
                    "topk_weights": ";".join(f"{w:.6f}" for _, w in top_pairs),
                })
            if (start // args.batch_size) % 20 == 0:
                print(f"Processed {min(start + args.batch_size, len(ds))}/{len(ds)}")

if __name__ == '__main__':
    main()
