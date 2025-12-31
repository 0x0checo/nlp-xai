# Evaluating Faithfulness of Text Explanation Methods for BERT-based Sentiment Classification

This repository contains the materials for a small empirical study on **explanation faithfulness** for neural text classifiers. Concretely, it compares **Integrated Gradients (IG)**, **attention weights**, and **LIME** on a **BERT-base-uncased** sentiment classifier fine-tuned on **SST-2**, and evaluates how well each method identifies tokens the model *actually relies on*. 

## Motivation

Explanation methods are widely used in NLP, but an explanation can look intuitive while still failing to reflect a model’s true decision process. This project focuses on **faithfulness**—whether removing “important” tokens (as identified by an explainer) truly decreases the model’s confidence.

## Task & Dataset

* **Task:** sentence-level **binary sentiment classification**
* **Dataset:** **SST-2** from GLUE (movie review sentences)

  * Train: **67,349**
  * Validation: **872**
  * Test: **1,821**
* All faithfulness analyses are conducted on the **validation set** to keep the test set unused. 

## Model

* **Base model:** BERT-base-uncased
* **Fine-tuning:** 3 epochs, learning rate **2e-5**, max sequence length **128**
* **Validation accuracy:** **92.5%**

## Explanation Methods

This study evaluates three popular explanation approaches on the *same fine-tuned classifier* for a fair comparison: 

1. **Integrated Gradients (IG)**

   * Uses **50 integration steps** with a **PAD-token baseline**. 
2. **Attention Weights**

   * Uses the final-layer attention distribution from **[CLS] → token**, **averaged across heads**. 
3. **LIME**

   * Perturbation-based local surrogate model; run with **500 perturbation samples per sentence**.

## Faithfulness Evaluation: Deletion Test

Faithfulness is measured with a **deletion test**:

1. Rank tokens by an explanation method.
2. Remove the **top-k** tokens from the input.
3. Recompute the model probability for the **original predicted label**.
4. Record the **probability drop**:
   [
   \Delta p_k = p_{\text{orig}} - p_{\text{after deletion}}
   ]
   Larger (\Delta p_k) means higher faithfulness. Experiments use (k \in {1, 3, 5, 10}).

**Deletion granularity note:** for IG and attention, deletion is at the **BERT token/subword** level; for LIME, it is at the **word** level in raw text (avoids subword fragmentation). 

## Results

### Mean probability drop (higher is better)

| Method    |        k=1 |        k=3 |        k=5 |       k=10 |
| --------- | ---------: | ---------: | ---------: | ---------: |
| IG        |     0.1520 |     0.2624 |     0.3110 |     0.4072 |
| Attention |     0.1117 |     0.1880 |     0.2362 |     0.3410 |
| LIME      | **0.2304** | **0.3524** | **0.4007** | **0.4338** |

Overall ranking is consistent: **LIME > IG > Attention**.

### Statistical significance (Wilcoxon signed-rank)

A Wilcoxon signed-rank test is used to check whether LIME’s gains are robust. LIME significantly outperforms both IG and attention across all tested k values (p < 0.05), with especially strong significance vs attention (p < 0.001).

## Qualitative Example

A small case study illustrates *why* the methods behave differently. For example, in a positive sentence (“it’s a charming and often affecting journey”), LIME highlights sentiment-bearing adjectives (“affecting”, “charming”), while attention may assign high weight to punctuation or stopwords—consistent with the “Attention is not Explanation” line of work.

## Compute / Runtime

Experiments were run on **Google Colab (T4 GPU)** to keep LIME efficient, with total execution time under ~20 minutes. 

## Limitations

* Only **one model (BERT-base)** and **one dataset (SST-2)** are tested; results may differ for other tasks/models.
* LIME’s perturbation sampling introduces randomness; results should be interpreted statistically rather than as an absolute guarantee. 
* Deletion is only an approximation of “removing a feature” (subword vs word deletion may affect fluency and measured drops). 

## Reference

If you build on this work, please cite the accompanying report in this repo:

* *Evaluating Faithfulness of Text Explanation Methods for BERT-based Sentiment Classification* 
