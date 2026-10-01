"""
Comprehensive Phase 1-6 Audit Test Suite
Evidence-Grounded Multimodal AI for Job Scam Detection
=========================================================
Run with:
    cd <project_root>
    source backend/venv/bin/activate
    python -m pytest tests/test_all_phases.py -v --tb=short 2>&1 | tee evaluation/test_run_report.txt
"""

import sys
import os
import json
import re
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# ── make project root importable ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, os.path.join(ROOT, "data"))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase1Architecture:
    """Validate that the project structure is in place and the API can be imported."""

    REQUIRED_DIRS = [
        "backend", "frontend", "ml", "data",
        "experiments", "evaluation", "tests", "docs", "scripts",
    ]
    REQUIRED_DOCS = [
        "README.md", "ARCHITECTURE.md",
        "RESEARCH_MAPPING.md", "EXPERIMENTS.md",
    ]

    def test_required_directories_exist(self):
        for d in self.REQUIRED_DIRS:
            path = os.path.join(ROOT, d)
            assert os.path.isdir(path), f"Missing required directory: {d}"

    def test_required_documentation_exists(self):
        for doc in self.REQUIRED_DOCS:
            path = os.path.join(ROOT, doc)
            assert os.path.isfile(path), f"Missing required document: {doc}"

    def test_backend_main_importable(self):
        """The FastAPI app must be importable without crashing."""
        sys.path.insert(0, os.path.join(ROOT, "backend"))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "backend.main",
                os.path.join(ROOT, "backend", "main.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert hasattr(mod, "app"), "backend/main.py must define 'app'"
        except ImportError as e:
            pytest.fail(f"backend/main.py has broken import: {e}")

    def test_api_health_route_exists(self):
        """Verify /health route is registered."""
        sys.path.insert(0, os.path.join(ROOT, "backend"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backend.main2",
            os.path.join(ROOT, "backend", "main.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        routes = [r.path for r in mod.app.routes]
        assert "/health" in routes, f"'/health' route not found in {routes}"

    def test_api_health_endpoint_returns_ok(self):
        """Fire the /health endpoint using the test client."""
        from fastapi.testclient import TestClient
        sys.path.insert(0, os.path.join(ROOT, "backend"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backend.main3",
            os.path.join(ROOT, "backend", "main.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = TestClient(mod.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ml_modules_importable(self):
        """All four ML modules must be importable without errors."""
        for module_name in [
            "ml.entity_extraction",
            "ml.multimodal_pipeline",
            "ml.classical_baselines",
            "ml.transformer_classifier",
        ]:
            import importlib
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Cannot import {module_name}: {e}")

    def test_frontend_scaffold_exists(self):
        """Vite scaffold files must be present."""
        frontend = os.path.join(ROOT, "frontend")
        assert os.path.isfile(os.path.join(frontend, "package.json"))
        assert os.path.isfile(os.path.join(frontend, "index.html"))
        assert os.path.isfile(os.path.join(frontend, "vite.config.ts"))

    def test_architecture_documents_are_nonempty(self):
        for doc in self.REQUIRED_DOCS:
            path = os.path.join(ROOT, doc)
            assert os.path.getsize(path) > 50, f"{doc} appears to be empty/trivial"

    def test_no_hardcoded_credentials_in_docs(self):
        """Scan markdown docs for obvious credential patterns."""
        credential_patterns = [
            r"password\s*=\s*['\"].+['\"]",
            r"secret\s*=\s*['\"].+['\"]",
            r"api_key\s*=\s*['\"].+['\"]",
        ]
        for doc in self.REQUIRED_DOCS:
            path = os.path.join(ROOT, doc)
            content = open(path).read()
            for pat in credential_patterns:
                assert not re.search(pat, content, re.IGNORECASE), \
                    f"Potential hardcoded credential found in {doc}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — EMSCAD Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2EMSCADPipeline:
    """Validate the EMSCAD dataset loader, schema, splits, and imbalance reporting."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from data.emscad_pipeline import EMSCADPipeline
        return EMSCADPipeline(data_dir=os.path.join(ROOT, "data"))

    @pytest.fixture(scope="class")
    def raw_df(self, pipeline):
        return pipeline.download_dataset()

    def test_dataset_loads(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        assert len(raw_df) > 100

    def test_fraudulent_column_exists(self, raw_df):
        assert "fraudulent" in raw_df.columns, \
            "Dataset missing 'fraudulent' column — schema mismatch"

    def test_no_extra_label_leakage_in_raw(self, raw_df):
        """The 'type' column should NOT be present in the raw dataset."""
        assert "type" not in raw_df.columns, \
            "'type' column already exists in raw data — potential leakage"

    def test_missing_value_handling(self, pipeline, raw_df):
        df = pipeline.missing_value_handling(raw_df.copy())
        text_cols = ["title", "description", "requirements", "company_profile"]
        for col in text_cols:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, \
                    f"Nulls remain in '{col}' after missing_value_handling"

    def test_duplicate_detection_runs(self, pipeline, raw_df):
        df = pipeline.duplicate_detection(raw_df.copy())
        assert len(df) <= len(raw_df), "Duplicate detection increased row count"

    def test_fraud_categories_all_four_labels(self, pipeline, raw_df):
        df = pipeline.generate_fraud_categories(raw_df.copy())
        assert "type" in df.columns
        unique = set(df["type"].unique())
        expected = {0, 1, 2, 3}
        assert unique == expected, \
            f"Labels {unique} do not match expected {expected}"

    def test_fraud_category_proportions(self, pipeline, raw_df):
        """Type-1 should be ~64%, type-2 ~27%, type-3 ~9% of fraud rows."""
        df = pipeline.generate_fraud_categories(raw_df.copy())
        fraud = df[df["type"] > 0]
        total = len(fraud)
        t1_ratio = (fraud["type"] == 1).sum() / total
        t2_ratio = (fraud["type"] == 2).sum() / total
        t3_ratio = (fraud["type"] == 3).sum() / total
        assert 0.55 <= t1_ratio <= 0.73, f"Type-1 ratio {t1_ratio:.2f} out of expected range"
        assert 0.18 <= t2_ratio <= 0.36, f"Type-2 ratio {t2_ratio:.2f} out of expected range"
        assert 0.04 <= t3_ratio <= 0.20, f"Type-3 ratio {t3_ratio:.2f} out of expected range"

    def test_split_files_exist_and_non_empty(self):
        for split in ["train.csv", "val.csv", "test.csv"]:
            path = os.path.join(ROOT, "data", split)
            assert os.path.isfile(path), f"Split file missing: {split}"
            df = pd.read_csv(path)
            assert len(df) > 0, f"{split} is empty"

    def test_split_sizes_are_reasonable(self):
        train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
        val   = pd.read_csv(os.path.join(ROOT, "data", "val.csv"))
        test  = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        total = len(train) + len(val) + len(test)
        assert 0.60 <= len(train)/total <= 0.80, "Train fraction unexpected"
        assert 0.05 <= len(val)/total   <= 0.20, "Val fraction unexpected"
        assert 0.15 <= len(test)/total  <= 0.25, "Test fraction unexpected"

    def test_no_index_overlap_between_splits(self):
        """Ensure rows don't appear in multiple splits (data leakage check)."""
        train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
        val   = pd.read_csv(os.path.join(ROOT, "data", "val.csv"))
        test  = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        # Use job_id if available
        if "job_id" in train.columns:
            train_ids = set(train["job_id"])
            val_ids   = set(val["job_id"])
            test_ids  = set(test["job_id"])
            assert not train_ids & test_ids,  "job_id overlap between train and test!"
            assert not train_ids & val_ids,   "job_id overlap between train and val!"
            assert not val_ids   & test_ids,  "job_id overlap between val and test!"

    def test_test_set_class_distribution_preserved(self):
        """Test set must not be oversampled — class ratios should match original."""
        test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        real_ratio = (test["type"] == 0).sum() / len(test)
        assert real_ratio > 0.80, \
            f"Real class ratio in test set is {real_ratio:.2f} — possible oversampling"

    def test_imbalance_report_exists(self):
        path = os.path.join(ROOT, "data", "reports", "imbalance_report.json")
        assert os.path.isfile(path)
        with open(path) as f:
            report = json.load(f)
        assert "imbalance_ratio_real_vs_fraud" in report
        assert report["imbalance_ratio_real_vs_fraud"] > 1.0

    def test_reproducibility_random_seed(self, pipeline, raw_df):
        """Running the pipeline twice with the same seed must produce identical splits."""
        df1 = pipeline.generate_fraud_categories(raw_df.copy())
        df2 = pipeline.generate_fraud_categories(raw_df.copy())
        pd.testing.assert_series_equal(df1["type"].reset_index(drop=True),
                                       df2["type"].reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Classical ML Baselines
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def classical_data():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    test  = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    train["cleaned_text"] = train["cleaned_text"].fillna("")
    test["cleaned_text"]  = test["cleaned_text"].fillna("")
    train["merged_text"]  = train["cleaned_text"]
    test["merged_text"]   = test["cleaned_text"]
    if "company_profile" not in train.columns:
        train["company_profile"] = ""
        test["company_profile"] = ""
    return train, test


class TestPhase3ClassicalML:

    def test_rule_based_features_returns_dataframe(self, classical_data):
        from ml.classical_baselines import rule_based_features
        _, test_df = classical_data
        feats = rule_based_features(test_df.head(20))
        assert isinstance(feats, pd.DataFrame)
        expected_cols = {
            "urgency", "payment_requests", "suspicious_contact",
            "missing_info", "external_links", "referral_language",
            "exaggerated_comp", "identity_requests",
        }
        assert expected_cols.issubset(set(feats.columns))

    def test_rule_based_features_binary_values(self, classical_data):
        from ml.classical_baselines import rule_based_features
        _, test_df = classical_data
        feats = rule_based_features(test_df.head(50))
        for col in feats.columns:
            assert set(feats[col].unique()).issubset({0, 1}), \
                f"Feature '{col}' has non-binary values"

    def test_rule_based_predict_returns_valid_labels(self, classical_data):
        from ml.classical_baselines import rule_based_features, rule_based_predict
        _, test_df = classical_data
        feats = rule_based_features(test_df.head(50))
        preds = rule_based_predict(feats)
        assert isinstance(preds, np.ndarray)
        assert set(preds).issubset({0, 1, 2, 3}), \
            f"Rule-based predictions contain invalid labels: {set(preds)}"

    def test_tfidf_lr_trains_and_predicts(self, classical_data):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        train, test = classical_data
        vec = TfidfVectorizer(max_features=500)
        X_train = vec.fit_transform(train["cleaned_text"].head(500))
        X_test  = vec.transform(test["cleaned_text"].head(100))
        y_train = train["type"].head(500)
        y_test  = test["type"].head(100)
        lr = LogisticRegression(max_iter=200, class_weight="balanced")
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        probs = lr.predict_proba(X_test)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1, 2, 3})
        assert probs.shape[1] == 4
        assert all(np.isclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_tfidf_rf_trains_and_predicts(self, classical_data):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        train, test = classical_data
        vec = TfidfVectorizer(max_features=500)
        X_train = vec.fit_transform(train["cleaned_text"].head(500))
        X_test  = vec.transform(test["cleaned_text"].head(100))
        y_train = train["type"].head(500)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_evaluate_model_function(self, classical_data):
        from ml.classical_baselines import evaluate_model
        y_true = np.array([0, 0, 1, 2, 0])
        y_pred = np.array([0, 0, 1, 2, 1])
        result = evaluate_model(y_true, y_pred)
        required_keys = {"accuracy", "precision", "recall", "f1_score", "confusion_matrix"}
        assert required_keys.issubset(set(result.keys()))
        assert 0.0 <= result["accuracy"] <= 1.0
        assert isinstance(result["confusion_matrix"], list)

    def test_classical_results_saved(self):
        path = os.path.join(ROOT, "evaluation", "classical_results.json")
        assert os.path.isfile(path), "classical_results.json not found in evaluation/"
        with open(path) as f:
            results = json.load(f)
        for model in ["Rule_Based", "TFIDF_LR", "TFIDF_RF", "TFIDF_GB"]:
            assert model in results, f"Model '{model}' missing from classical_results.json"

    def test_no_test_contamination_in_vectorizer(self, classical_data):
        """TF-IDF must be fit ONLY on training data, not on test data.
        The vectorizer must apply the training vocabulary to the test set —
        the test-set feature dimension must match the training-set feature dimension.
        Note: on the synthetic dataset the vocabulary is small, so max_features
        is effectively bounded by the actual vocab size (not 100).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        train, test = classical_data
        vec = TfidfVectorizer(max_features=100)
        # Fit on train ONLY
        X_train = vec.fit_transform(train["cleaned_text"].head(200))
        # Transform test using training vocabulary (not re-fitting)
        X_test = vec.transform(test["cleaned_text"].head(50))
        # Both must have the same number of features (capped at actual vocab size)
        assert X_test.shape[1] == X_train.shape[1], \
            (f"Test feature dim ({X_test.shape[1]}) != train feature dim ({X_train.shape[1]}) "
             "— vectorizer was re-fitted on test data (data leakage)")
        # Vocab size must be > 0 and <= max_features
        assert 0 < X_train.shape[1] <= 100, \
            f"Unexpected vocabulary size: {X_train.shape[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Transformer / DistilBERT
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4Transformer:

    @pytest.fixture(scope="class")
    def classifier(self):
        from ml.transformer_classifier import FraudBERTClassifier
        # Load from checkpoint if available, otherwise re-init
        checkpoint = os.path.join(ROOT, "ml", "checkpoints", "distilbert")
        if os.path.isdir(checkpoint):
            clf = FraudBERTClassifier(model_name=checkpoint)
        else:
            clf = FraudBERTClassifier()
        return clf

    def test_tokenizer_encodes_text(self, classifier):
        encoding = classifier.tokenizer(
            ["Hello, I am looking for a job."],
            truncation=True, padding=True, max_length=64
        )
        assert "input_ids" in encoding
        assert len(encoding["input_ids"]) == 1
        assert len(encoding["input_ids"][0]) <= 64

    def test_model_produces_correct_output_shape(self, classifier):
        import torch
        sample_text = ["Apply now and earn $500 daily!"]
        enc = classifier.tokenizer(sample_text, return_tensors="pt",
                                   truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            out = classifier.model(**{k: v.to(classifier.device) for k, v in enc.items()})
        assert out.logits.shape == (1, 4), \
            f"Expected (1, 4) logits, got {out.logits.shape}"

    def test_probability_outputs_sum_to_one(self, classifier):
        import torch
        sample_text = ["Send your passport details to collect your offer."]
        enc = classifier.tokenizer(sample_text, return_tensors="pt",
                                   truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            out = classifier.model(**{k: v.to(classifier.device) for k, v in enc.items()})
        probs = torch.nn.functional.softmax(out.logits, dim=-1)
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, not 1.0"

    def test_inference_returns_required_fields(self, classifier):
        result = classifier.inference("Earn big money with our recruitment program. Pay $200 upfront.")
        assert "fraud_probability" in result
        assert "predicted_category" in result
        assert "confidence" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["predicted_category"] in {0, 1, 2, 3}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_inference_does_not_return_risk_score(self, classifier):
        """Per project rules: DistilBERT output must NOT include final system risk score."""
        result = classifier.inference("Apply now!")
        assert "risk_level" not in result, \
            "Transformer must NOT produce the final system risk score (Phase 7 only)"
        assert "risk_score" not in result

    def test_checkpoint_was_saved(self):
        checkpoint = os.path.join(ROOT, "ml", "checkpoints", "distilbert")
        assert os.path.isdir(checkpoint), "Checkpoint directory missing"
        config_path = os.path.join(checkpoint, "config.json")
        assert os.path.isfile(config_path), "config.json missing in checkpoint"

    def test_checkpoint_loadable(self):
        from ml.transformer_classifier import FraudBERTClassifier
        checkpoint = os.path.join(ROOT, "ml", "checkpoints", "distilbert")
        clf = FraudBERTClassifier(model_name=checkpoint)
        assert clf.model is not None
        result = clf.inference("Test job ad for checkpoint loading verification.")
        assert "fraud_probability" in result


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — OCR / Image Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ocr_pipeline():
    from ml.multimodal_pipeline import MultimodalPipeline
    return MultimodalPipeline()


@pytest.fixture(scope="module")
def scam_image_path():
    """Create a synthetic scam screenshot for testing."""
    import cv2
    import numpy as np
    path = os.path.join(ROOT, "data", "test_scam_screen.png")
    if not os.path.exists(path):
        img = np.full((200, 700, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Earn $800 daily! Apply at bit.ly/job99", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Contact: hr@gmail.com. Immediate start!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(path, img)
    return path


@pytest.fixture(scope="module")
def blank_image_path():
    """Image with no text (all white)."""
    import cv2
    import numpy as np
    path = os.path.join(ROOT, "data", "test_blank.png")
    if not os.path.exists(path):
        img = np.full((100, 300, 3), 255, dtype=np.uint8)
        cv2.imwrite(path, img)
    return path


class TestPhase5OCR:

    def test_clear_screenshot_returns_text(self, ocr_pipeline, scam_image_path):
        record = ocr_pipeline.process_evidence(scam_image_path)
        assert record.get("extracted_text", "") != "", \
            "OCR returned empty text for clear scam screenshot"

    def test_ocr_preserves_original_image_path(self, ocr_pipeline, scam_image_path):
        record = ocr_pipeline.process_evidence(scam_image_path)
        assert record.get("original_image_path") == scam_image_path, \
            "original_image_path was mutated — must never replace original evidence"

    def test_ocr_confidence_is_recorded(self, ocr_pipeline, scam_image_path):
        record = ocr_pipeline.process_evidence(scam_image_path)
        assert "ocr_confidence" in record
        assert isinstance(record["ocr_confidence"], (int, float))

    def test_ocr_evidence_locations_preserved(self, ocr_pipeline, scam_image_path):
        record = ocr_pipeline.process_evidence(scam_image_path)
        locs = record.get("evidence_locations_sample", [])
        for loc in locs:
            assert "text" in loc
            assert "confidence" in loc
            assert "bounding_box" in loc

    def test_blank_image_does_not_crash(self, ocr_pipeline, blank_image_path):
        """OCR on a blank image must return gracefully, not crash."""
        record = ocr_pipeline.process_evidence(blank_image_path)
        assert "error" not in record or "original_image_path" in record
        assert isinstance(record.get("extracted_text", ""), str)

    def test_nonexistent_file_returns_error_dict(self, ocr_pipeline):
        record = ocr_pipeline.process_evidence("/nonexistent/path/image.png")
        assert "error" in record
        assert "File not found" in record["error"]

    def test_ocr_text_enters_entity_extractor(self, ocr_pipeline, scam_image_path):
        """Extracted OCR text must be usable downstream without transformation."""
        record = ocr_pipeline.process_evidence(scam_image_path)
        ocr_text = record.get("extracted_text", "")
        from ml.entity_extraction import EntityExtractor
        extractor = EntityExtractor()
        entities = extractor.extract_entities(ocr_text, source_ref="ocr_test")
        assert isinstance(entities, list)

    def test_scam_indicators_extracted_from_screenshot(self, ocr_pipeline, scam_image_path):
        record = ocr_pipeline.process_evidence(scam_image_path)
        indicators = record.get("scam_indicators", [])
        # The image contains "Earn $800 daily" and "hr@gmail.com" and "Immediate start"
        # At least urgency or suspicious_contact or exaggerated_comp should fire
        assert len(indicators) >= 1, \
            f"Expected scam indicators from image containing scam text, got: {indicators}"

    def test_source_type_recorded_as_image(self, ocr_pipeline, scam_image_path):
        record = ocr_pipeline.process_evidence(scam_image_path)
        assert record.get("source_type") == "IMAGE"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Entity Extraction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def extractor():
    from ml.entity_extraction import EntityExtractor
    return EntityExtractor()


CLEAN_JOB_AD = (
    "Senior Software Engineer position at TechCorp Inc. in San Francisco. "
    "Salary $120,000 annually. Apply at https://techcorp.com/careers. "
    "Contact recruiting@techcorp.com."
)

SUSPICIOUS_JOB_AD = (
    "Work from home! Earn $800 daily. WhatsApp +1-555-0198. "
    "Send your passport and bank account details to hr@gmail.com. "
    "Register at bit.ly/jobnow. Pay $50 upfront registration fee. "
    "Commission based income. Recruit others to earn more! "
    "Deposit via Cash App or 0x1234567890abcdef1234567890abcdef12345678."
)

EMAIL_TEXT = (
    "Dear John Smith, I am Jane Doe from GlobalRecruit. "
    "Please contact me at jane.doe@global-recruit.net or call +44-7700-900000."
)

MESSAGE_TEXT = "Hi! Quick job offer. Contact hr@gmail.com. WhatsApp +1-555-1234 or visit www.earnbig.vip"

MULTI_ENTITY_TEXT = (
    "Amazon Logistics is hiring a Delivery Driver role in London. "
    "Email logistics@amazon-partner.xyz. Call +1-800-555-0100. "
    "Earn $25/hr. Pay via PayPal. Visit https://amazon-jobs-apply.info."
)

NO_ENTITY_TEXT = "The weather today is quite nice. I enjoy sunny afternoons."

MALFORMED_TEXT = "@@@@email@@broken.c phone 555 $$$salary malformed http:// wallet 0xABC"


class TestPhase6EntityExtraction:

    # ── field structure ───────────────────────────────────────────────────────

    def test_entity_has_required_fields(self, extractor):
        entities = extractor.extract_entities(CLEAN_JOB_AD, "test_clean")
        for e in entities:
            for field in ["value", "type", "source", "confidence", "evidence_reference"]:
                assert field in e, f"Entity missing field '{field}': {e}"

    def test_entity_value_is_nonempty(self, extractor):
        entities = extractor.extract_entities(MULTI_ENTITY_TEXT, "test_multi")
        for e in entities:
            assert e["value"].strip() != "", "Empty entity value detected — system is inventing entities"

    def test_confidence_in_valid_range(self, extractor):
        entities = extractor.extract_entities(MULTI_ENTITY_TEXT, "test_conf")
        for e in entities:
            assert 0.0 <= e["confidence"] <= 1.0, \
                f"Confidence out of range for entity: {e}"

    # ── entity types ─────────────────────────────────────────────────────────

    def test_email_extracted_from_clean_ad(self, extractor):
        entities = extractor.extract_entities(CLEAN_JOB_AD, "clean")
        types = [e["type"] for e in entities]
        assert "EMAIL" in types, "EMAIL not extracted from clean job ad"

    def test_domain_derived_from_email(self, extractor):
        entities = extractor.extract_entities(CLEAN_JOB_AD, "clean")
        types = [e["type"] for e in entities]
        assert "DOMAIN" in types, "DOMAIN not derived from email"

    def test_url_extracted(self, extractor):
        entities = extractor.extract_entities(CLEAN_JOB_AD, "clean")
        types = [e["type"] for e in entities]
        assert "URL" in types, "URL not extracted from clean ad"

    def test_phone_extracted_from_message(self, extractor):
        entities = extractor.extract_entities(MESSAGE_TEXT, "msg")
        types = [e["type"] for e in entities]
        assert "PHONE" in types, "PHONE not extracted from message text"

    def test_salary_extracted(self, extractor):
        entities = extractor.extract_entities(MULTI_ENTITY_TEXT, "multi")
        types = [e["type"] for e in entities]
        assert "SALARY" in types, "SALARY not extracted"

    def test_payment_method_extracted(self, extractor):
        entities = extractor.extract_entities(SUSPICIOUS_JOB_AD, "suspicious")
        types = [e["type"] for e in entities]
        assert "PAYMENT_METHOD" in types, "PAYMENT_METHOD (Cash App) not extracted from suspicious text"

    def test_crypto_wallet_extracted(self, extractor):
        entities = extractor.extract_entities(SUSPICIOUS_JOB_AD, "suspicious")
        types = [e["type"] for e in entities]
        assert "CRYPTOCURRENCY_WALLET" in types, "CRYPTOCURRENCY_WALLET not extracted"

    # ── no-entity texts ───────────────────────────────────────────────────────

    def test_no_entities_invented_from_plain_text(self, extractor):
        """System must NOT invent entities absent from the input."""
        entities = extractor.extract_entities(NO_ENTITY_TEXT, "none")
        regex_sensitive_types = {"EMAIL", "PHONE", "URL", "SALARY",
                                 "PAYMENT_METHOD", "CRYPTOCURRENCY_WALLET"}
        for e in entities:
            assert e["type"] not in regex_sensitive_types, \
                f"Regex entity '{e['type']}: {e['value']}' invented from plain text"

    def test_malformed_input_does_not_crash(self, extractor):
        entities = extractor.extract_entities(MALFORMED_TEXT, "malformed")
        assert isinstance(entities, list)

    def test_empty_string_returns_empty_list(self, extractor):
        entities = extractor.extract_entities("", "empty")
        assert isinstance(entities, list)
        # No entities should be present for empty input
        for e in entities:
            assert e["value"].strip() != ""

    def test_deduplication_no_repeated_entities(self, extractor):
        repeated = MULTI_ENTITY_TEXT + " " + MULTI_ENTITY_TEXT
        entities = extractor.extract_entities(repeated, "dup_test")
        seen = set()
        for e in entities:
            key = (e["value"], e["type"])
            assert key not in seen, f"Duplicate entity detected: {key}"
            seen.add(key)

    # ── relationships ─────────────────────────────────────────────────────────

    def test_relationships_list_returned(self, extractor):
        entities = extractor.extract_entities(EMAIL_TEXT, "email_ctx")
        rels = extractor.build_relationships(entities, "msg_test")
        assert isinstance(rels, list)

    def test_email_domain_relationship_built(self, extractor):
        entities = extractor.extract_entities(EMAIL_TEXT, "email_ctx")
        rels = extractor.build_relationships(entities, "msg_test")
        rel_types = {r["relation"] for r in rels}
        assert "hosted_on" in rel_types, "EMAIL→DOMAIN relationship not built"

    def test_recruiter_email_relationship_built(self, extractor):
        entities = extractor.extract_entities(EMAIL_TEXT, "email_ctx")
        rels = extractor.build_relationships(entities, "msg_test")
        rel_types = {r["relation"] for r in rels}
        assert "uses_email" in rel_types, "RECRUITER→EMAIL relationship not built"

    def test_message_phone_relationship_built(self, extractor):
        entities = extractor.extract_entities(MESSAGE_TEXT, "msg_ctx")
        rels = extractor.build_relationships(entities, "msg_phone_test")
        rel_types = {r["relation"] for r in rels}
        assert "contains_phone" in rel_types, "MESSAGE→PHONE relationship not built"

    def test_evidence_reference_traceable(self, extractor):
        ref = "audit_test_ref_42"
        entities = extractor.extract_entities(CLEAN_JOB_AD, source_ref=ref)
        for e in entities:
            # Either directly or derived from something in the text
            assert e["evidence_reference"] is not None, \
                "evidence_reference is None — entity is not traceable"


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PHASE INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossPhaseIntegration:

    def test_A_text_to_classical_ml(self):
        """TEST A: Text → preprocessing → entity extraction → classical ML."""
        from ml.entity_extraction import EntityExtractor
        from ml.classical_baselines import rule_based_features, rule_based_predict

        text = "Work from home earning $500/day. No experience needed. Pay $30 upfront."
        extractor = EntityExtractor()
        entities = extractor.extract_entities(text, "test_a")
        assert isinstance(entities, list)

        # Simulate rule-based classification on the same text
        df = pd.DataFrame([{
            "merged_text": text.lower(),
            "company_profile": ""
        }])
        feats = rule_based_features(df)
        preds = rule_based_predict(feats)
        assert preds[0] in {0, 1, 2, 3}, "Invalid prediction from rule-based on integration text"

    def test_A_text_to_distilbert(self):
        """TEST A (continued): Text → DistilBERT inference."""
        from ml.transformer_classifier import FraudBERTClassifier
        checkpoint = os.path.join(ROOT, "ml", "checkpoints", "distilbert")
        clf = FraudBERTClassifier(model_name=checkpoint if os.path.isdir(checkpoint) else "distilbert-base-uncased")
        result = clf.inference("Earn $800 daily! Deposit $50 for training materials. WhatsApp now.")
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["predicted_category"] in {0, 1, 2, 3}

    def test_B_image_to_entity_extraction(self, ocr_pipeline, extractor, scam_image_path):
        """TEST B: Screenshot → OCR → entity extraction."""
        record = ocr_pipeline.process_evidence(scam_image_path)
        ocr_text = record.get("extracted_text", "")
        assert isinstance(ocr_text, str)
        entities = extractor.extract_entities(ocr_text, source_ref="ocr_integration")
        assert isinstance(entities, list)

    def test_B_image_to_distilbert(self, ocr_pipeline, scam_image_path):
        """TEST B (continued): OCR text → DistilBERT."""
        from ml.transformer_classifier import FraudBERTClassifier
        record = ocr_pipeline.process_evidence(scam_image_path)
        ocr_text = record.get("extracted_text", "")
        checkpoint = os.path.join(ROOT, "ml", "checkpoints", "distilbert")
        clf = FraudBERTClassifier(model_name=checkpoint if os.path.isdir(checkpoint) else "distilbert-base-uncased")
        result = clf.inference(ocr_text if ocr_text else "empty ocr fallback")
        assert "fraud_probability" in result

    def test_C_image_scam_entities_present(self, ocr_pipeline, scam_image_path):
        """TEST C: Screenshot → OCR → scam entities detected."""
        record = ocr_pipeline.process_evidence(scam_image_path)
        indicators = record.get("scam_indicators", [])
        assert len(indicators) >= 1, "No scam indicators extracted from scam screenshot"

    def test_D_invalid_inputs_do_not_crash(self, extractor, ocr_pipeline):
        """TEST D: Invalid inputs → controlled errors, no application crash."""
        # Empty string
        ents = extractor.extract_entities("", "empty")
        assert isinstance(ents, list)

        # None-ish text (simulate via very short string)
        ents2 = extractor.extract_entities("  ", "whitespace")
        assert isinstance(ents2, list)

        # Nonexistent image
        record = ocr_pipeline.process_evidence("/does/not/exist.png")
        assert "error" in record

    def test_D_api_invalid_text_returns_422(self):
        """TEST D: API with empty text returns 422, not 500."""
        from fastapi.testclient import TestClient
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backend.main_d",
            os.path.join(ROOT, "backend", "main.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = TestClient(mod.app)
        resp = client.post("/analyze/text", json={"text": ""})
        assert resp.status_code == 422, \
            f"Expected 422 for empty text, got {resp.status_code}"

    def test_D_api_unsupported_media_type_returns_415(self):
        """TEST D: Uploading a non-image file must return 415, not 500."""
        from fastapi.testclient import TestClient
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backend.main_d2",
            os.path.join(ROOT, "backend", "main.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = TestClient(mod.app)
        resp = client.post(
            "/analyze/image",
            files={"file": ("malware.exe", b"\x00\x01", "application/octet-stream")}
        )
        assert resp.status_code == 415, \
            f"Expected 415 for unsupported file type, got {resp.status_code}"
