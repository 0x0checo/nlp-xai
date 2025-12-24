import argparse
import os
import csv
import torch
from datasets import load_dataset
from IG_Explainer import IGExplainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default='/proj/uppmax2020-2-2/wenyuli/outputs/ckpt_sst2_bert/checkpoint-6315')
    ap.add_argument("--split", default="validation",
                    choices=["train", "validation", "test"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--out_csv", default="/proj/uppmax2020-2-2/wenyuli/outputs/explainability_sst2/sst2_val_ig.csv")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:{device}')
    # Load dataset
    dataset = load_dataset('glue', 'sst2')
    ds_split = dataset[args.split]
    print(f"Split: {args.split}, #examples: {len(ds_split)}")

    # Build explainer
    explainer = IGExplainer(
        ckpt_dir=args.ckpt_dir,
        max_length=args.max_length,
        device=device
    )

    # Prepare output
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "text",
                "pred_label",
                "pred_prob",
                "topk_tokens",
                "topk_scores",
            ],
        )
        writer.writeheader()

        total = len(ds_split)
        for idx in range(total):
            row = ds_split[idx]
            text = row['sentence']

            # Get IG for one sentence
            df_top, pred_label, pred_prob=explainer.explain(
                text,
                topk=args.topk,
                n_steps=args.n_steps,
            )

            tokens = ";".join(df_top["token"].tolist())
            scores = ";".join(f"{s:.6f}" for s in df_top["ig_score"].tolist())

            writer.writerow(
                {
                    "id": idx,
                    "text": text,
                    "pred_label": pred_label,
                    "pred_prob": f"{pred_prob:.6f}",
                    "topk_tokens": tokens,
                    "topk_scores": scores,
                }
            )

            if (idx + 1) % 20 == 0 or idx == total - 1:
                print(f"Processed {idx + 1}/{total}")

    print("Done. Saved to:", args.out_csv)

if __name__ == '__main__':
    main()

