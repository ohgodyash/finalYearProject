# Research Mapping

This document explicitly maps the theoretical components from the reference papers to their concrete implementation in this project.

- **Naudé 2023**
  → classical ML + fraud categories + feature engineering
  (Base for baseline models and empirical rule-set based features. Support for real, identity theft, corporate identity theft, and MLM fraud categories.)

- **Fraud-BERT 2025**
  → transformer text classifier
  (Base for the implementation of the DistilBERT model for contextual fake-job text classification, handling highly imbalanced data.)

- **Anansi 2026**
  → multimodal messages + OCR + infrastructure signals
  (Base for analyzing message-based job scams via OCR, extracting domains, phone numbers, and identifying infrastructure reuse patterns.)

- **2026 Survey**
  → Text → Image → OCR → Entities → Verification → Consistency → Fusion
  (Base for the overarching evidence-grounded multimodal verification pipeline and logistic regression evidence fusion into risk probability.)
