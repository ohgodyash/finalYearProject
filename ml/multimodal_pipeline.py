import cv2
import pytesseract
import spacy
import re
import os
import json
import numpy as np
from PIL import Image

class MultimodalPipeline:
    def __init__(self):
        print("Initializing Multimodal Pipeline (Anansi-inspired)...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if download hasn't finished, though it should have
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Regex patterns for specific entities not well captured by default NER
        self.patterns = {
            "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "PHONE": r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            "URL": r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
        }
        
        # Scam indicators (Rule-based)
        self.indicator_rules = {
            "urgency": r'urgent|immediate start|act now|hurry',
            "payment_request": r'investment|deposit|upfront fee|registration fee|pay to',
            "suspicious_contact": r'@yahoo|@gmail|@hotmail|whatsapp|telegram',
            "external_links": r'apply at|click here|bit\.ly|tinyurl',
            "referral_language": r'referral|commission based|downline|recruit others',
            "exaggerated_comp": r'earn \$[5-9]\d{2,}|unlimited income|get rich|guaranteed profit',
            "identity_request": r'social security|passport|bank account|routing number'
        }

    def preprocess_image(self, image_path):
        """
        Preprocesses image for better OCR accuracy.
        Converts to grayscale and applies thresholding.
        """
        # Read image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Save temp image for pytesseract
        temp_path = "temp_processed_image.png"
        cv2.imwrite(temp_path, thresh)
        return temp_path

    def run_ocr(self, image_path):
        """
        Runs Tesseract OCR.
        Preserves original text, confidence, and bounding box locations.
        """
        # We also want bounding boxes and confidences
        # Use image_to_data which returns a dict
        try:
            ocr_data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"OCR Error: Ensure Tesseract is installed. {e}")
            return "", [], 0.0
            
        extracted_text = []
        evidence_locations = []
        confidences = []
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            if int(float(conf)) > 0 and text:
                extracted_text.append(text)
                confidences.append(conf)
                evidence_locations.append({
                    "text": text,
                    "confidence": conf,
                    "bounding_box": {
                        "x": ocr_data['left'][i],
                        "y": ocr_data['top'][i],
                        "w": ocr_data['width'][i],
                        "h": ocr_data['height'][i]
                    }
                })
                
        full_text = " ".join(extracted_text)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, evidence_locations, avg_conf

    def extract_entities(self, text):
        """
        Uses spaCy and Regex to extract named entities.
        """
        doc = self.nlp(text)
        entities = {"ORG": [], "PERSON": [], "MONEY": [], "GPE": []}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
                    
        # Custom Regex Extraction
        for label, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            entities[label] = list(set(matches))
            
        return entities

    def extract_scam_indicators(self, text):
        """
        Maps text to specific rule-based scam indicators.
        """
        indicators = []
        text_lower = text.lower()
        for indicator_name, pattern in self.indicator_rules.items():
            if re.search(pattern, text_lower):
                indicators.append(indicator_name)
        return indicators

    def process_evidence(self, image_path):
        """
        End-to-end processing of a multimodal input.
        Never replaces the original evidence.
        """
        if not os.path.exists(image_path):
            return {"error": "File not found."}
            
        # Preprocess (optional, we'll just use raw for now to preserve original state)
        # Using raw image preserves original evidence
        
        full_text, evidence_locations, avg_conf = self.run_ocr(image_path)
        
        entities = self.extract_entities(full_text)
        indicators = self.extract_scam_indicators(full_text)
        
        # Construct the final evidence record
        evidence_record = {
            "source_type": "IMAGE",
            "original_image_path": image_path,
            "ocr_confidence": avg_conf,
            "extracted_text": full_text,
            "entities": entities,
            "scam_indicators": indicators,
            "evidence_locations_sample": evidence_locations[:5] # Truncated for display
        }
        
        return evidence_record

def main():
    pipeline = MultimodalPipeline()
    
    # Create a dummy image to test if one doesn't exist
    test_img = "data/dummy_scam.png"
    if not os.path.exists(test_img):
        img = np.zeros((200, 600, 3), dtype=np.uint8)
        img.fill(255) # White background
        cv2.putText(img, "Earn $500 daily! WhatsApp me.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Contact hr@gmail.com. Immediate start.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_img, img)
        
    print(f"Processing test image: {test_img}")
    record = pipeline.process_evidence(test_img)
    
    # Save the output to demonstrate evidence extraction
    out_path = "evaluation/multimodal_extraction.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=4)
        
    print(f"Extraction complete. Results saved to {out_path}")
    print(json.dumps(record, indent=4))

if __name__ == "__main__":
    main()
