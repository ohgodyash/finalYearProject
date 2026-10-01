import spacy
import re
import json

class EntityExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.regex_patterns = {
            "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "PHONE": r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            "URL": r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
            "SALARY": r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:per hour|/hr|a day|/day|daily|weekly|monthly|annually|a year|/yr)',
            "PAYMENT_METHOD": r'(?i)\b(paypal|cash app|venmo|zelle|strike|wire transfer|direct deposit)\b',
            "CRYPTOCURRENCY_WALLET": r'(?i)\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59})\b'
        }

    def _extract_domain_from_email(self, email):
        if "@" in email:
            return email.split("@")[1]
        return None

    def _extract_domain_from_url(self, url):
        # Basic extraction
        domain = re.sub(r'^https?://', '', url)
        domain = re.sub(r'^www\.', '', domain)
        domain = domain.split('/')[0]
        return domain

    def extract_entities(self, text, source_ref="text_document"):
        doc = self.nlp(text)
        entities = []
        
        def add_entity(value, ent_type, confidence, source, ref):
            # Do not invent missing
            if not value or not value.strip():
                return
            # Deduplicate by value and type
            for e in entities:
                if e['value'] == value and e['type'] == ent_type:
                    return
            entities.append({
                "value": value,
                "type": ent_type,
                "source": source,
                "confidence": confidence,
                "evidence_reference": ref
            })

        # 1. SpaCy Extraction
        for ent in doc.ents:
            if ent.label_ == "ORG":
                add_entity(ent.text, "COMPANY", 0.85, "spacy_ner", source_ref)
            elif ent.label_ == "PERSON":
                add_entity(ent.text, "RECRUITER", 0.80, "spacy_ner", source_ref)
            elif ent.label_ == "GPE" or ent.label_ == "LOC":
                add_entity(ent.text, "LOCATION", 0.90, "spacy_ner", source_ref)

        # Basic heuristic for JOB_TITLE (looks for title case phrases before words like 'position' or 'role')
        job_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3})\s+(?:position|role|job|opportunity)', text)
        if job_match:
            add_entity(job_match.group(1), "JOB_TITLE", 0.70, "regex_heuristic", source_ref)

        # 2. Regex Extraction
        for ent_type, pattern in self.regex_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                add_entity(match.group(0), ent_type, 0.95, "regex", source_ref)

        # Derived Entities
        for e in list(entities):
            if e['type'] == 'EMAIL':
                domain = self._extract_domain_from_email(e['value'])
                if domain:
                    add_entity(domain, "DOMAIN", 0.99, "derived_from_email", e['value'])
            elif e['type'] == 'URL':
                domain = self._extract_domain_from_url(e['value'])
                if domain:
                    add_entity(domain, "DOMAIN", 0.99, "derived_from_url", e['value'])

        return entities

    def build_relationships(self, entities, source_message_id):
        relationships = []
        
        def add_rel(source_val, source_type, target_val, target_type, relation):
            relationships.append({
                "source": {"value": source_val, "type": source_type},
                "target": {"value": target_val, "type": target_type},
                "relation": relation
            })

        # Separate entities by type
        by_type = {}
        for e in entities:
            by_type.setdefault(e['type'], []).append(e)

        # MESSAGE -> PHONE, MESSAGE -> URL
        for phone in by_type.get('PHONE', []):
            add_rel(source_message_id, "MESSAGE", phone['value'], "PHONE", "contains_phone")
            
        for url in by_type.get('URL', []):
            add_rel(source_message_id, "MESSAGE", url['value'], "URL", "contains_url")

        # RECRUITER -> EMAIL (link if both exist in the same context)
        recruiters = by_type.get('RECRUITER', [])
        emails = by_type.get('EMAIL', [])
        if recruiters and emails:
            # Simple heuristic: link first recruiter to first email
            add_rel(recruiters[0]['value'], "RECRUITER", emails[0]['value'], "EMAIL", "uses_email")

        # EMAIL -> DOMAIN
        for email in emails:
            domain = self._extract_domain_from_email(email['value'])
            if domain:
                add_rel(email['value'], "EMAIL", domain, "DOMAIN", "hosted_on")

        # COMPANY -> DOMAIN
        companies = by_type.get('COMPANY', [])
        domains = by_type.get('DOMAIN', [])
        if companies and domains:
            add_rel(companies[0]['value'], "COMPANY", domains[0]['value'], "DOMAIN", "owns_domain")

        # JOB -> COMPANY
        jobs = by_type.get('JOB_TITLE', [])
        if jobs and companies:
            add_rel(jobs[0]['value'], "JOB_TITLE", companies[0]['value'], "COMPANY", "belongs_to")

        return relationships

def main():
    extractor = EntityExtractor()
    sample_text = (
        "Hello, I am John Doe from Amazon Recruiting. We are offering a Data Analyst role "
        "in New York. You can earn $500 daily. Please send your resume to hr@amazon-jobs.vip. "
        "Also message our WhatsApp at +1-555-0198. Deposit to 0x1234567890abcdef1234567890abcdef12345678 via Cash App."
    )
    
    print("Extracting Entities...")
    entities = extractor.extract_entities(sample_text, source_ref="msg_101")
    print(json.dumps(entities, indent=4))
    
    print("\nBuilding Relationships...")
    relations = extractor.build_relationships(entities, source_message_id="msg_101")
    print(json.dumps(relations, indent=4))

if __name__ == "__main__":
    main()
