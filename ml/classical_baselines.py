import pandas as pd
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def rule_based_features(df):
    """
    Extracts interpretable rule-based features from text.
    """
    features = pd.DataFrame(index=df.index)
    
    # Text context
    merged = df['merged_text'].fillna('').str.lower()
    
    # Urgency
    features['urgency'] = merged.str.contains(r'urgent|immediate start|act now|hurry', regex=True).astype(int)
    
    # Payment requests
    features['payment_requests'] = merged.str.contains(r'investment|deposit|upfront fee|registration fee|pay to', regex=True).astype(int)
    
    # Suspicious contact information (e.g., using personal emails or weird phone formats)
    features['suspicious_contact'] = merged.str.contains(r'@yahoo|@gmail|@hotmail|\[email\]', regex=True).astype(int)
    
    # Missing information (if company profile is empty)
    features['missing_info'] = (df['company_profile'] == "").astype(int)
    
    # External application links
    features['external_links'] = merged.str.contains(r'apply at|click here|bit\.ly|tinyurl', regex=True).astype(int)
    
    # Referral/commission language
    features['referral_language'] = merged.str.contains(r'referral|commission based|downline|recruit others', regex=True).astype(int)
    
    # Exaggerated compensation
    features['exaggerated_comp'] = merged.str.contains(r'earn \$[5-9]\d{2,}|unlimited income|get rich|guaranteed profit', regex=True).astype(int)
    
    # Identity-related requests
    features['identity_requests'] = merged.str.contains(r'social security|passport|bank account|routing number', regex=True).astype(int)
    
    return features

def rule_based_predict(features):
    """
    Simple heuristic logic to map rules to fraud categories.
    0=REAL, 1=IDENTITY_THEFT, 2=CORPORATE_IDENTITY_THEFT, 3=MULTI_LEVEL_MARKETING
    """
    preds = []
    for _, row in features.iterrows():
        if row['referral_language'] > 0:
            preds.append(3) # MLM
        elif row['identity_requests'] > 0:
            preds.append(1) # Identity theft
        elif row['payment_requests'] > 0 or row['exaggerated_comp'] > 0 or row['missing_info'] > 0:
            preds.append(2) # Corp identity theft (fallback for generic scams here)
        else:
            preds.append(0) # Real
    return np.array(preds)

def evaluate_model(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            pass # Fails if not all classes are present in y_true
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def main():
    data_dir = "data"
    eval_dir = "evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Ensure no NaNs in cleaned_text
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    
    X_train_text = train_df['cleaned_text']
    y_train = train_df['type']
    X_test_text = test_df['cleaned_text']
    y_test = test_df['type']
    
    # Re-create merged_text for rule-based model since it's not in the saved CSV
    # Actually, we can just use cleaned_text, but we'll try to extract features from it.
    train_df['merged_text'] = train_df['cleaned_text']
    test_df['merged_text'] = test_df['cleaned_text']
    # And mock missing_info for test
    train_df['company_profile'] = train_df.get('company_profile', "")
    test_df['company_profile'] = test_df.get('company_profile', "")
    
    results = {}
    
    print("Evaluating Rule-Based Classifier...")
    test_rules = rule_based_features(test_df)
    y_pred_rules = rule_based_predict(test_rules)
    # Rule based has no prob output
    results["Rule_Based"] = evaluate_model(y_test, y_pred_rules)
    
    print("Vectorizing Text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_vec, y_train)
    results["TFIDF_LR"] = evaluate_model(y_test, lr.predict(X_test_vec), lr.predict_proba(X_test_vec))
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_vec, y_train)
    results["TFIDF_RF"] = evaluate_model(y_test, rf.predict(X_test_vec), rf.predict_proba(X_test_vec))
    
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_vec, y_train)
    results["TFIDF_GB"] = evaluate_model(y_test, gb.predict(X_test_vec), gb.predict_proba(X_test_vec))
    
    # Save JSON results
    with open(os.path.join(eval_dir, "classical_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Comparison Table
    md_content = "# Classical ML Baselines Comparison\n\n"
    md_content += "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
    md_content += "|-------|----------|-----------|--------|----------|---------|\n"
    for model, m_res in results.items():
        roc = f"{m_res['roc_auc']:.4f}" if m_res['roc_auc'] else "N/A"
        md_content += f"| {model} | {m_res['accuracy']:.4f} | {m_res['precision']:.4f} | {m_res['recall']:.4f} | {m_res['f1_score']:.4f} | {roc} |\n"
        
    with open(os.path.join(eval_dir, "CLASSICAL_COMPARISON.md"), "w") as f:
        f.write(md_content)
        
    print("Classical baselines evaluation complete. Results saved in evaluation/.")

if __name__ == "__main__":
    main()
