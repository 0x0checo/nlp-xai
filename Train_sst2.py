import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import evaluate
import numpy as np
import torch
import wandb

def main():
    # Set random seeds for pytorch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Load dataset
    dataset = load_dataset('glue', 'sst2')
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)

    # Define tokenizer function
    def tokenize(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            max_length=args.max_length
        )
    # Apply toknize to all datasets
    encoded = dataset.map(tokenize, batched=True)
    # Load evaluation metrics
    metric = evaluate.load('glue', 'sst2')

    # Define a evaluation metric function
    def compute_metrics(eval_pred):
        # Unpack eval_pred
        logits, labels = eval_pred
        # Convert logits to idx
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt,
        num_labels=2
    )

    # Set trainingargs
    train_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs_train,
        per_device_eval_batch_size=args.bs_eval,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=2e-5,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to=['wandb']
    )

    # Set trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=encoded['train'],
        eval_dataset=encoded['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start training
    if args.do_train:
        print('Start training...')
        trainer.train()
        # Save checkpoints and tokenizer
        trainer.save_model(args.ckpt_dir)
        tokenizer.save_pretrained(args.ckpt_dir)
    # If training complete
    print('Done!')

# Run main function
if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("--model_ckpt", default="bert-base-uncased")
  ap.add_argument("--ckpt_dir", default="ckpt_sst2_bert")
  ap.add_argument("--epochs", type=int, default=3)
  ap.add_argument("--bs_train", type=int, default=32)
  ap.add_argument("--bs_eval", type=int, default=64)
  ap.add_argument("--max_length", type=int, default=128)
  ap.add_argument("--do_train", action="store_true")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  main(args)
