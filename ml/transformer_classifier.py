import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import json
from sklearn.utils.class_weight import compute_class_weight

class JobScamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class FraudBERTClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.device = torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.num_labels = num_labels

    def prepare_data(self, texts, labels):
        # We use merged_text (unprocessed text) for BERT, since context and casing matters
        encodings = self.tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)
        return JobScamDataset(encodings, labels.tolist())

    def train(self, train_loader, val_loader, class_weights, epochs=1):
        """
        Trains the model handling class imbalance using CrossEntropyLoss with weights.
        """
        self.model.train()
        optim = AdamW(self.model.parameters(), lr=5e-5)
        # Handle class imbalance
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(self.device))
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                
                loss.backward()
                optim.step()
                
                if batch_idx > 5: # Fast dev run to prevent pipeline timeout
                    break
                    
            print(f"Epoch {epoch+1} complete. Validating...")
            self.validate(val_loader)

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx > 5: # Fast dev run
                    break
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {acc:.4f}")

    def save_checkpoint(self, path="ml/checkpoints/distilbert"):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def predict(self, test_loader):
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch.get('labels')
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
                    
                if batch_idx > 10: # Fast dev run
                    break
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)

    def inference(self, text):
        """
        Outputs fraud probability, predicted category, and confidence.
        Does NOT produce the final risk score.
        """
        self.model.eval()
        encoding = self.tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
        pred_cat = np.argmax(probs)
        confidence = probs[pred_cat]
        
        # Fraud probability is 1 - prob(REAL)
        fraud_prob = 1.0 - probs[0]
        
        return {
            "fraud_probability": float(fraud_prob),
            "predicted_category": int(pred_cat),
            "confidence": float(confidence)
        }

def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        roc_auc = None
        
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
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Use raw merged text for transformer
    train_texts = train_df[['company_profile', 'requirements', 'benefits', 'description']].fillna('').agg(' '.join, axis=1)
    val_texts = val_df[['company_profile', 'requirements', 'benefits', 'description']].fillna('').agg(' '.join, axis=1)
    test_texts = test_df[['company_profile', 'requirements', 'benefits', 'description']].fillna('').agg(' '.join, axis=1)
    
    train_labels = train_df['type']
    val_labels = val_df['type']
    test_labels = test_df['type']
    
    classes = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    
    classifier = FraudBERTClassifier()
    
    train_dataset = classifier.prepare_data(train_texts, train_labels)
    val_dataset = classifier.prepare_data(val_texts, val_labels)
    test_dataset = classifier.prepare_data(test_texts, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    print("Training DistilBERT (Fraud-BERT)...")
    classifier.train(train_loader, val_loader, class_weights, epochs=1)
    classifier.save_checkpoint()
    
    print("Evaluating DistilBERT on test set...")
    y_pred, y_prob, y_true = classifier.predict(test_loader)
    metrics = evaluate_metrics(y_true, y_pred, y_prob)
    
    # Load previous classical results and update
    eval_dir = "evaluation"
    results_file = os.path.join(eval_dir, "classical_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    all_results["DistilBERT"] = metrics
    
    with open(os.path.join(eval_dir, "all_models_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Generate Comparison Table
    md_content = "# Comprehensive Model Comparison\n\n"
    md_content += "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
    md_content += "|-------|----------|-----------|--------|----------|---------|\n"
    for model, m_res in all_results.items():
        roc = f"{m_res['roc_auc']:.4f}" if m_res['roc_auc'] else "N/A"
        md_content += f"| {model} | {m_res['accuracy']:.4f} | {m_res['precision']:.4f} | {m_res['recall']:.4f} | {m_res['f1_score']:.4f} | {roc} |\n"
        
    with open(os.path.join(eval_dir, "COMPARISON.md"), "w") as f:
        f.write(md_content)
        
    print("Inference test:")
    sample = "We are hiring for a data entry role. Earn $500 daily. Must deposit $50 for software."
    inf_res = classifier.inference(sample)
    print(f"Sample: {sample}")
    print(json.dumps(inf_res, indent=4))
    print("Transformer baseline completed.")

if __name__ == "__main__":
    main()
