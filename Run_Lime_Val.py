import argparse
import csv
import os
import torch
from datasets import load_dataset
from lime_explainer_colab import LimeSST2Explainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        required=True
    )
    ap.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"]
    )
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument(
        "--num_samples",
        type=int,
        default=500
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        required=True
    )
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading dataset SST2...')
    dataset = load_dataset('glue', 'sst2')
    ds = dataset[args.split]
    n = len(ds)
    print(f"Loaded split={args.split}, #examples={n}")

    explainer = LimeSST2Explainer(
        ckpt_dir=args.ckpt_dir,
        max_length=args.max_length,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
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

        total = n if args.limit < 0 else min(args.limit, n)

        for i in range(total):
            row = ds[i]
            text = row['sentence']

            df, pred_label, pred_prob = explainer.explain(
                text=text,
                topk=args.topk,
                num_samples=args.num_samples,
            )

            tokens = ";".join(df["token"].tolist())
            scores = ";".join(f"{s:.6f}" for s in df["lime_score"].tolist())
            writer.writerow(
                {
                    "id": i,
                    "text": text,
                    "pred_label": pred_label,
                    "pred_prob": f"{pred_prob:.6f}",
                    "topk_tokens": tokens,
                    "topk_scores": scores,
                }
            )

            if (i + 1) % 20 == 0 or i == total - 1:
                print(f"Processed {i + 1}/{total}")

    print("Done. Saved to:", args.out_csv)


if __name__ == "__main__":
    main()
