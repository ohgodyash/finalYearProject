# Research Experiments

This project includes a built-in evaluation module to perform comparative research experiments, as requested.

## Models Evaluated

- **Baseline 1:** Rule-based classifier (derived from Naudé 2023)
- **Baseline 2:** TF-IDF + Logistic Regression
- **Baseline 3:** TF-IDF + Random Forest
- **Baseline 4:** TF-IDF + Gradient Boosting
- **Model 5:** DistilBERT text classifier (derived from Fraud-BERT 2025)
- **Model 6 (Final):** Multimodal evidence-fusion model (DistilBERT + rules + verification + consistency + infrastructure features → Logistic Regression fusion)

## Metrics

Because the data is highly imbalanced, relying on accuracy alone is insufficient. The evaluation module assesses:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Class-wise performance
