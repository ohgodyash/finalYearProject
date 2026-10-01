"""
FastAPI Backend — Evidence-Grounded Multimodal Job Scam Detection
Implements health endpoint + core analysis routes.
All ML inference is read-only against saved artefacts.
"""
import os
import sys
import json
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(
    title="Job Scam Detector API",
    description="Evidence-Grounded Multimodal AI for Job Scam Detection",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class TextAnalysisRequest(BaseModel):
    text: str
    source_ref: Optional[str] = "api_text_input"


class AnalysisResponse(BaseModel):
    status: str
    entities: list
    relationships: list
    scam_indicators: list


# ── lazy-loaded singletons ────────────────────────────────────────────────────
_extractor = None
_multimodal = None


def get_extractor():
    global _extractor
    if _extractor is None:
        from ml.entity_extraction import EntityExtractor
        _extractor = EntityExtractor()
    return _extractor


def get_multimodal():
    global _multimodal
    if _multimodal is None:
        from ml.multimodal_pipeline import MultimodalPipeline
        _multimodal = MultimodalPipeline()
    return _multimodal


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint — confirms the API is running."""
    return {"status": "ok", "service": "job-scam-detector", "version": "0.1.0"}


@app.get("/")
def root():
    return {"message": "Job Scam Detector API. See /docs for the API reference."}


@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalysisRequest):
    """
    Accepts raw job-ad text or message text.
    Returns extracted entities, relationships, and scam indicators.
    Does NOT produce the final risk score (that is Phase 7 — Evidence Fusion).
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text field must not be empty")

    extractor = get_extractor()
    entities = extractor.extract_entities(request.text, source_ref=request.source_ref)
    relationships = extractor.build_relationships(entities, source_message_id=request.source_ref)

    # scam indicators (reuse multimodal indicator logic on text directly)
    import re
    indicator_rules = {
        "urgency": r'urgent|immediate start|act now|hurry',
        "payment_request": r'investment|deposit|upfront fee|registration fee|pay to',
        "suspicious_contact": r'@yahoo|@gmail|@hotmail|whatsapp|telegram',
        "referral_language": r'referral|commission based|downline|recruit others',
        "exaggerated_comp": r'earn \$[5-9]\d{2,}|unlimited income|get rich|guaranteed profit',
        "identity_request": r'social security|passport|bank account|routing number',
    }
    indicators = [k for k, pat in indicator_rules.items() if re.search(pat, request.text.lower())]

    return AnalysisResponse(
        status="ok",
        entities=entities,
        relationships=relationships,
        scam_indicators=indicators,
    )


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an image upload (screenshot, offer letter, etc.).
    Runs OCR then entity/indicator extraction.
    The original file bytes are preserved; only the extracted text is processed further.
    """
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Accepted: {allowed_types}",
        )

    # Save to a temp location (we keep the original bytes intact)
    temp_dir = "data/uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename or "upload.png")

    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    pipeline = get_multimodal()
    record = pipeline.process_evidence(temp_path)

    # Also run structured entity extraction on OCR text
    ocr_text = record.get("extracted_text", "")
    extractor = get_extractor()
    entities = extractor.extract_entities(ocr_text, source_ref=f"ocr:{file.filename}")
    relationships = extractor.build_relationships(entities, source_message_id=f"img:{file.filename}")

    record["structured_entities"] = entities
    record["relationships"] = relationships
    return record
