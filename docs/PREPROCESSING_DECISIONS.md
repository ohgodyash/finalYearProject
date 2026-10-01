# Preprocessing Decisions

This document outlines the preprocessing methodology applied to the EMSCAD dataset in accordance with the reference research papers (primarily Naudé 2023 and Fraud-BERT 2025).

## 1. Missing Value Handling
**Decision:** Missing values in text fields (`company_profile`, `description`, `requirements`, `benefits`) were replaced with empty strings (`""`).
**Reasoning:** NLP models (both TF-IDF and Transformers) require text input. Dropping rows with missing profiles would discard valuable data, as missing company profiles are in themselves a strong signal of fraudulent intent (as identified in the rule-based baselines).

## 2. Duplicate Detection
**Decision:** Exact duplicates across core text fields (`title`, `company_profile`, `description`, `requirements`) are dropped.
**Reasoning:** To prevent data leakage between train/val/test splits and to prevent the model from over-indexing on heavily spammed identical posts during training.

## 3. Anonymization Preservation
**Decision:** PII (emails and phone numbers) are masked using regex (`[EMAIL]`, `[PHONE]`).
**Reasoning:** EMSCAD is largely anonymized, but applying explicit regex masks ensures GDPR compliance and forces the models to learn linguistic patterns of fraud rather than overfitting to specific scammer phone numbers or email addresses (which are ephemeral).

## 4. Class Label Generation
**Decision:** The binary `fraudulent` label is expanded into four classes (`REAL`, `IDENTITY_THEFT`, `CORPORATE_IDENTITY_THEFT`, `MULTI_LEVEL_MARKETING`). Since the original annotations by Naudé are not public, we simulated the distribution (64% Type 1, 27% Type 2, 9% Type 3) to enable multiclass evaluation.
**Reasoning:** Required to reproduce the research methodology and evaluate multiclass performance.

## 5. Text Merging and Normalization
**Decision:** For classical ML baselines, the four main text fields are merged into a single `merged_text` column, lowercased, stripped of numbers/punctuation, tokenized, and stop-words removed.
**Reasoning:** This precisely replicates the Bag-of-Words and TF-IDF preprocessing steps described in Naudé 2023. Note: The DistilBERT model (Phase 2) will use raw text with its own tokenizer, as Transformers rely on punctuation and casing for context.

## 6. Train / Validation / Test Split
**Decision:** 70% Train, 10% Validation, 20% Test split using stratification.
**Reasoning:** Stratification is critical due to the extreme class imbalance (~95% Real, 5% Fraud). 
**CRITICAL:** The test set is completely isolated and is **NOT** subjected to any oversampling (like SMOTE). Oversampling the test set invalidates the evaluation (a flaw noted in previous works).
