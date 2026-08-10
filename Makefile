# ApexRAG Makefile — convenience targets for development workflow
# Usage: make <target>

.PHONY: help install test eval server train-router drift clean

help:
	@echo ""
	@echo "  ApexRAG — Zero-Cost Enterprise RAG Harness"
	@echo "  ==========================================="
	@echo ""
	@echo "  make install        Install all dependencies in virtual environment"
	@echo "  make test           Run full pytest test suite"
	@echo "  make eval           Run end-to-end evaluation pipeline"
	@echo "  make server         Launch FastAPI production server (port 8000)"
	@echo "  make train-router   Train dynamic strategy router ML model"
	@echo "  make drift          Run embedding drift detection manually"
	@echo "  make clean          Remove cached build artifacts"
	@echo ""

install:
	python -m venv venv
	.\venv\Scripts\pip install --upgrade pip
	.\venv\Scripts\pip install -r requirements.txt
	@echo "✓ Installation complete. Activate: .\\venv\\Scripts\\activate"

test:
	.\venv\Scripts\python -m pytest tests/ -v --tb=short

eval:
	.\venv\Scripts\python scripts/run_full_pipeline.py

server:
	.\venv\Scripts\uvicorn src.api.main:app --reload --port 8001

train-router:
	.\venv\Scripts\python scripts/generate_qa_key.py
	.\venv\Scripts\python scripts/train_router.py

drift:
	.\venv\Scripts\python -m src.drift.check_drift

clean:
	if exist data\chroma_eval_db rmdir /s /q data\chroma_eval_db
	if exist data\chroma_db rmdir /s /q data\chroma_db
	if exist __pycache__ rmdir /s /q __pycache__
	for /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "✓ Clean complete."
