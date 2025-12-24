# Project Proposal
## Wenyu Li
### October 5, 2025

## 1 Purpose and Aims

With the rapid development of Large Language Models(LLMs), more and more natural language processing(NLP) tasks are greatly enhanced, achieving a very high level of performance in areas such as text classification. However, these advanced models also raise a lot of concerns about their transparency and interpretability. Ranging from researchers to common public users, they need not only accurate predictions but they also need an understanding of why these predictions can be trusted. So the aim of this project is to compare and evaluate two major approaches of explainable AI (XAI) in text classification: gradient-based approaches (e.g., Integrated Gradients) and example-based approaches (e.g., LIME). Specifically, the project seeks to:

- Investigate the pros and cons of two popular explainers in text classification.
- Conduct systematic evaluations of explanations using faithfulness metric.
- Explore the feasibility of attention weights as a baseline interpretability role.

By achieving these goals, the project will provide insights into which methods are more reliable in practice and under what conditions.

## 2 Survey of the Field

Explainable AI (XAI) has become an increasingly important research area, because large language models (LLMs) have achieved unprecedented results in natural language processing tasks such as text classification. Despite their success, these models are often criticized for being "black boxes," providing limited transparency into how predictions are made (Lipton, 2018). This lack of interpretability raises concerns about model reliability and trustworthiness, especially in high-risk fields like healthcare, finance, and legal decision-making.

Several approaches have been proposed to enhance the interpretability of the models. Gradient-based methods, such as Integrated Gradients (Sundararajan et al., 2017), assign importance to input tokens by integrating gradients of the output. These methods are faithful to the internal computations of the model, but can be sensitive to input perturbations and non-linearities of the model (Mudrakarta et al., 2018). While these methods are often easier to understand, they can be computationally expensive and not always produce reliable explanations.

In contrast, example-based methods estimate the importance of input features by adding perturbations to the input data and observe how these changes affect predictions. LIME (Ribeiro et al., 2016), a novel explanation technique, helps users to understand a model’s predictions by estimating its behavior locally with a simpler model, like linear regression.

Some researchers have argued attention mechanisms as a potential source of interpretability, given that attention weights can influence model predictions (Jain and Wallace, 2019). However, other subsequent work has questioned the reliability of attention as an explanation, arguing that attention may not always correlate with model inference (Serrano and Smith, 2019; Wiegreffe and Pinter, 2019).

How to evaluate the explanation of a model is still very challenging. Some simple techniques can be set to test the faithfulness, like deletion or replacement: removing top-k ranked tokens should lead to prediction changes if the explanations are faithful (Atanasova et al., 2023).

Overall, while both gradient-based and example-based methods are widely used, their comparative performance in text classification tasks, particularly under systematic evaluation, leaves considerable room for further investigation.

## 3 Project Description

### 3.1 Theories and Methods

This project will explore the interpretability of Large Language Models (LLMs) in text classification by evaluating two main approaches: gradient-based (e.g., Integrated Gradients) and example-based approaches (e.g., LIME). Both approaches aim to reveal which parts of the input data contribute most to a model’s predictions although they are based on different theoretical mechanisms and computation process.

Gradient-based methods (such as Integrated Gradients) compute attribution scores by integrating gradients of the output with respect to the input features (Sundararajan et al., 2017). These methods are grounded in the theoretical basis that gradients imply how sensitively the output depends on each input token. By directly connecting a model’s predictions to its internal computation process, they provide an evidencing approach to measure the faithfulness of the decision-making process. However, the disadvantages of these methods are that they are very sensitive to small perturbations and heavily depend on the architecture of the model, which raise some concerns about their interpretability in practice (Mudrakarta et al., 2018).

In contrast, example-based methods estimate a model’s decision by sampling perturbed inputs and building a local simple model around instances to be explained, like LIME (Ribeiro et al., 2016). These methods are model-agnostic, meaning they can apply to all kinds of classifiers regardless of their architectures. But they also face some challenges: they heavily rely on the quality of sampling and they can be very computationally expensive, especially when handling long input sequences.

To further investigate above issues, this project will explore the role of attention mechanisms as a potential interpretability baseline. Attention weights inherently reveal how the model focuses on each token when making a prediction. Although some researchers suggest that attention weights can be used as an explanation (Wiegreffe and Pinter, 2019), there are some opposite arguments indicating that attention weights may not always align with true causal significance (Serrano and Smith, 2019; Wiegreffe and Pinter, 2019). Therefore, this project will use attention-based explanations as a comparative baseline, evaluating whether they correlate with more reliable attribution methods.

### 3.2 Experimental Design

The project will use a text classification task, where the interpretability of model predictions is clearly measurable. A pre-trained bert-based model will be fine-tuned on a benchmark dataset SST-2 from Hugging Face.

After model fine-tuning, three types of explanations will be generated for the same inputs:

1. Gradient-based explanations: using Integrated Gradients.
2. Example-based explanations: using LIME.
3. Attention-based explanations: derived from the model’s self-attention scores.

Each explanation method will output an attribution score for each input token, indicating its importance to the predicted label (positive or negative). These attribution scores will be evaluated by using faithfulness metrics, primarily deletion and insertion tests. In the deletion test, all tokens are ranked by their attribution scores and the most important tokens (e.g., top-3 tokens) are removed from the input, and the drop in model confidence is measured. A larger confidence decrease indicates higher faithfulness of explanations. In an insertion test, tokens are added back progressively, and the recovery confidence of the prediction is recorded. This framework can directly show the quantitative comparison between explanation methods.

All experiments will be conducted in Python, using Hugging Face Transformers, Captum (used for Integrated Gradients) and Lime library for explanation methods, with evaluation code implemented using PyTorch and scikit-learn. Results will be visualized through saliency heatmaps and importance ranking curves to conduct qualitative analysis.

### 3.3 Time Plan and Implementation

The project will be carried out over approximately two months and will be modified after peers’ review. So all processes will take about three months, divided into the following stages:

| Week | Task |
|------|------|
| 1–2  | Literature review, setup of experimental environment (model, dataset, and libraries) and prepare for presenting project proposal. |
| 3–4  | Fine-tuning a transformer model for text classification; Implementation of attention-based baseline. |
| 5–6  | Implementation of gradient-based and example-based explanation methods; generation of attribution maps. |
| 7–9  | Implementation of faithfulness evaluation metrics; Systematic evaluation (deletion/insertion tests, stability analysis, qualitative inspection); Complete the first version of project report. |
| 9–10 | Waiting for peers; review of the first version of project report. |
| 10–14| Writing of final report after getting peers’ review. |

The implementation will follow reproducible practices: fixed random seeds, detailed documentation, and open-source sharing of code and results.

## References

- Pepa Atanasova, Oana-Maria Camburu, Christina Lioma, Thomas Lukasiewicz, Jakob Grue Simonsen, and Isabelle Augenstein. Faithfulness tests for natural language explanations. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, Toronto, Canada, 2023.
- Sarthak Jain and Byron C. Wallace. Attention is not explanation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, pages 3543–3556, Minneapolis, USA, 2019.
- Zachary C. Lipton. The mythos of model interpretability. *Communications of the ACM*, 61(10):36–43, 2018.
- Pramod Kaushik Mudrakarta, Ankur Taly, Mukund Sundararajan, and Kedar Dhamdhere. Did the model understand the question? In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 1896–1906, Melbourne, Australia, 2018.
- Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. Why should I trust you? explaining the predictions of any classifier. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations*, pages 97–101, San Diego, California, 2016.
- Sofia Serrano and Noah A. Smith. Is attention interpretable? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 2931–2951, Florence, Italy, 2019.
- Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML)*, pages 3319–3328, Sydney, Australia, 2017.
- Sarah Wiegreffe and Yuval Pinter. Attention is not not explanation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 11–20, Hong Kong, China, 2019.
