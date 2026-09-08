"""
Invoice Processing Webhook Server
==================================
Cloud Run endpoint for the invoice-parser-v2 API.

Accepts:  POST /webhook/invoice-parser-v2  (binary PDF or base64 in body)
Returns:  { "invoices": [ { ...e-Invoice JSON... } ] }

Usage:
    python webhook_server.py                    # port 8080
    python webhook_server.py --port 9000        # custom port
    python webhook_server.py --skip-investigation  # faster, no compliance check
"""

import argparse
import base64
import json
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Resolve paths — same as agent.py
# ---------------------------------------------------------------------------
AGENT_PKG_DIR = Path(__file__).resolve().parent / "invoice_processing"
DATA_DIR = AGENT_PKG_DIR / "data"
EXEMPLARY_DIR = AGENT_PKG_DIR / "exemplary_data"
PROJECT_ROOT = AGENT_PKG_DIR.parent

sys.path.insert(0, str(AGENT_PKG_DIR.parent))

from invoice_processing.shared_libraries.acting.general_invoice_agent import (
    process_invoice,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Invoice Processing Webhook",
    description="GST e-Invoice extraction — Cloud Run API endpoint",
    version="1.0.0",
)

# Global flag (set via CLI)
SKIP_INVESTIGATION = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_case_id() -> str:
    """Generate a unique case ID for this request."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"webhook_{ts}_{short_uuid}"


def _save_pdf_to_case(pdf_bytes: bytes, case_id: str) -> Path:
    """Save uploaded PDF into exemplary_data/case_id/invoice.pdf."""
    case_dir = EXEMPLARY_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = case_dir / "invoice.pdf"
    pdf_path.write_bytes(pdf_bytes)
    return case_dir


def _cleanup_case(case_id: str):
    """Remove temporary case folders after processing."""
    for base in [EXEMPLARY_DIR, DATA_DIR / "agent_output", DATA_DIR / "alf_output"]:
        case_dir = base / case_id
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)


def _read_postprocessing(case_id: str) -> dict | None:
    """Read the Postprocessing_Data.json output."""
    # Check ALF output first (if ALF revised), then agent output
    for base in [DATA_DIR / "alf_output", DATA_DIR / "agent_output"]:
        pp_path = base / case_id / "Postprocessing_Data.json"
        if pp_path.exists():
            with open(pp_path, encoding="utf-8") as f:
                return json.load(f)
    return None


def _run_pipeline(case_id: str) -> dict:
    """Run the full Acting -> (Investigation) -> ALF pipeline and return result."""
    from invoice_processing.agent import run_inference

    skip = "true" if SKIP_INVESTIGATION else "false"
    result = run_inference(case_id, skip_investigation=skip)
    return result


def _build_response(
    case_id: str,
    pipeline_result: dict,
    start_time: float,
    raw_pdf_text: str | None = None,
    buyer_gstin_list: list[str] | None = None,
) -> dict:
    """Build the e-Invoice JSON response."""
    einvoice = _read_postprocessing(case_id)

    if einvoice is None:
        return {
            "status": "error",
            "errorCode": pipeline_result.get("status", "ERROR"),
            "message": pipeline_result.get("error", "Pipeline failed — no output produced"),
            "failed_stage": "PIPELINE",
            "transaction_id": case_id,
        }

    # Run post-extraction validation with raw text and buyer GSTIN list
    from invoice_processing.shared_libraries.acting.general_invoice_agent import (
        validate_and_enrich_einvoice,
    )
    _proc = einvoice.pop("_processing", None)
    einvoice = validate_and_enrich_einvoice(
        einvoice,
        raw_pdf_text=raw_pdf_text,
        buyer_gstin_list=buyer_gstin_list,
    )
    if _proc:
        einvoice["_processing"] = _proc

    # Strip _processing metadata into separate field
    processing_meta = einvoice.pop("_processing", {})

    # Wrap in invoices array
    return {
        "invoices": [einvoice],
        "_pipeline": {
            "transaction_id": case_id,
            "timestamp_start": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
            "pipeline_time_s": round(time.time() - start_time, 2),
            "acting_decision": pipeline_result.get("acting_decision", "ERROR"),
            "investigation_compliance": pipeline_result.get("investigation_compliance", "SKIPPED"),
            "investigation_score": pipeline_result.get("investigation_score", 0),
            "alf_revised": pipeline_result.get("alf_revised", False),
            "alf_rules_matched": pipeline_result.get("alf_rules_matched", 0),
            "status": pipeline_result.get("status", "ERROR"),
            **processing_meta,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/webhook/invoice-parser-v2")
async def invoice_parser_v2(request: Request):
    """Main webhook endpoint for invoice processing.

    Supports:
      1. Binary PDF upload (multipart/form-data with file field)
      2. Base64 PDF in JSON body  { "b64pdf": "..." } or { "file": "..." }
      3. Raw binary body (Content-Type: application/pdf)
    """
    start_time = time.time()
    case_id = _generate_case_id()
    pdf_bytes = None
    buyer_gstin_list = None

    content_type = request.headers.get("content-type", "")

    try:
        # --- Route 1: multipart/form-data (binary file upload) ---
        if "multipart/form-data" in content_type:
            form = await request.form()
            # Try common field names
            for field_name in ["pdf_file", "file", "data", "invoice"]:
                upload = form.get(field_name)
                if upload and hasattr(upload, "read"):
                    pdf_bytes = await upload.read()
                    break
            # If no named file found, try first file in form
            if pdf_bytes is None:
                for key, val in form.items():
                    if hasattr(val, "read"):
                        pdf_bytes = await val.read()
                        break
            # Extract buyerGstinList from form if present
            gstin_field = form.get("buyerGstinList")
            if gstin_field and isinstance(gstin_field, str):
                try:
                    buyer_gstin_list = json.loads(gstin_field)
                except Exception:
                    buyer_gstin_list = [g.strip() for g in gstin_field.split(",") if g.strip()]

        # --- Route 2: JSON body with base64 ---
        elif "application/json" in content_type:
            body = await request.json()
            b64_data = (
                body.get("b64pdf")
                or body.get("file")
                or body.get("data")
                or body.get("base64")
                or body.get("pdf")
                or body.get("b64string")
            )
            # Extract buyerGstinList from JSON body
            if not buyer_gstin_list:
                buyer_gstin_list = body.get("buyerGstinList")
            if b64_data:
                # Strip data URI prefix if present
                if "base64," in b64_data:
                    b64_data = b64_data.split("base64,", 1)[1]
                pdf_bytes = base64.b64decode(b64_data)

        # --- Route 3: Raw binary body ---
        elif "application/pdf" in content_type:
            pdf_bytes = await request.body()

        # --- Fallback: try reading raw body as base64 or binary ---
        if pdf_bytes is None:
            raw = await request.body()
            if raw:
                # Check if it's a PDF header
                if raw[:5] == b"%PDF-":
                    pdf_bytes = raw
                else:
                    # Try base64 decode
                    try:
                        decoded = base64.b64decode(raw)
                        if decoded[:5] == b"%PDF-":
                            pdf_bytes = decoded
                    except Exception:
                        pass

        if pdf_bytes is None or len(pdf_bytes) < 100:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "errorCode": "INPUT_NO_FILE",
                    "message": "No PDF file found. Send binary PDF, multipart upload, or base64 in JSON body.",
                    "transaction_id": case_id,
                },
            )

        # Validate it's a PDF
        if pdf_bytes[:5] != b"%PDF-":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "errorCode": "PDF_NOT_PDF",
                    "message": "Uploaded file is not a valid PDF.",
                    "transaction_id": case_id,
                },
            )

        # Save and process
        _save_pdf_to_case(pdf_bytes, case_id)
        pipeline_result = _run_pipeline(case_id)

        # Extract raw PDF text for regex enrichment
        raw_pdf_text = None
        try:
            import pdfplumber
            pdf_path = EXEMPLARY_DIR / case_id / "invoice.pdf"
            if pdf_path.exists():
                with pdfplumber.open(pdf_path) as pdf:
                    raw_pdf_text = "\n".join(
                        page.extract_text() or "" for page in pdf.pages[:5]
                    )
        except Exception:
            pass

        response = _build_response(
            case_id, pipeline_result, start_time,
            raw_pdf_text=raw_pdf_text,
            buyer_gstin_list=buyer_gstin_list,
        )

        # Determine HTTP status
        if "error" in response.get("status", ""):
            return JSONResponse(status_code=400, content=response)

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "errorCode": "PIPELINE_ERROR",
                "message": str(e),
                "transaction_id": case_id,
            },
        )
    finally:
        # Cleanup temp case files
        _cleanup_case(case_id)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "invoice-processing-webhook", "version": "1.0.0"}


@app.get("/")
async def root():
    """Root info endpoint."""
    return {
        "service": "Invoice Processing Webhook",
        "version": "1.0.0",
        "endpoints": {
            "POST /webhook/invoice-parser-v2": "Process invoice PDF, returns e-Invoice JSON",
            "GET /health": "Health check",
        },
        "accepts": [
            "multipart/form-data (binary PDF file)",
            "application/json (base64 PDF in body)",
            "application/pdf (raw binary PDF)",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice Processing Webhook Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument(
        "--skip-investigation", action="store_true",
        help="Skip Investigation stage (faster, Acting -> ALF only)",
    )
    args = parser.parse_args()

    SKIP_INVESTIGATION = args.skip_investigation

    print("=" * 60)
    print("INVOICE PROCESSING WEBHOOK SERVER")
    print("=" * 60)
    print(f"  Endpoint:  http://{args.host}:{args.port}/webhook/invoice-parser-v2")
    print(f"  Health:    http://{args.host}:{args.port}/health")
    print(f"  Skip investigation: {SKIP_INVESTIGATION}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
