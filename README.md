# Evidence-Grounded Multimodal AI for Job Scam Detection and Company Verification

This project is a research-based system derived from four core research papers on job scam detection. It implements a scalable, end-to-end multimodal pipeline.

## Overview
The system accepts job advertisement text, screenshots, offer letters, emails, and URLs, and performs:
1. Text classification
2. OCR extraction
3. Entity extraction
4. Rule-based scam detection
5. Company/domain verification
6. Cross-source consistency checking
7. Infrastructure analysis
8. Evidence collection
9. Evidence fusion
10. Risk assessment
11. Explainable report

## Architecture Overview
See `ARCHITECTURE.md` for a complete view of the pipeline.
See `RESEARCH_MAPPING.md` for how this relates to the base papers.
See `EXPERIMENTS.md` for the evaluation strategy.

## Phase 1 Structure
- `backend/` - FastAPI backend
- `frontend/` - React/Vite frontend
- `ml/` - ML models and evaluation
- `data/` - Dataset processing and mock data
- `experiments/` - Jupyter notebooks and experiment scripts
- `evaluation/` - Evaluation metrics and reports
- `tests/` - Pytest testing suite
- `docs/` - Project documentation
- `scripts/` - Utility scripts
# finalYearProject
