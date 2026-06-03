#!/usr/bin/env python3
"""
General Invoice Processing Multi-Agent System

Domain-independent invoice processing pipeline with configurable validation rules.
Each agent produces intermediate artifacts stored in output/{case_id}/

Version: 1.1.1
Reference: reconstructed_rules_book.md

Usage:
    python general_invoice_agent.py --base-dir path/to/invoices
    python general_invoice_agent.py --case path/to/single/case
    python general_invoice_agent.py --base-dir path/to/invoices --config config.json

Output:
    All outputs are saved in the SCRIPT DIRECTORY (not current working directory):
    general-invoice-processing-agent-gym/
    └── output/
        └── {case_id}/
            ├── 01_classification.json
            ├── 02_extraction.json
            ├── ...
            └── Postprocessing_Data.json

Note:
    - Outputs go to script's folder regardless of where you run it from
    - Config files are resolved relative to script directory if not absolute
    - .env file is loaded from script directory first
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

# Third-party imports
try:
    import pdfplumber
    from dotenv import load_dotenv
    from google.cloud import aiplatform
    from pydantic import BaseModel, Field
    from vertexai.generative_models import (
        GenerationConfig,
        GenerativeModel,
        Part,
    )
except ImportError:
    print("Error: Missing required package. Install with:")
    print(
        "pip install google-cloud-aiplatform pydantic pdfplumber python-dotenv"
    )
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Resolve paths: acting/ -> shared_libraries/ -> invoice_processing/ (package root with data/ inside)
SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_PKG_DIR = SCRIPT_DIR.parent.parent
OUTPUT_BASE_DIR = AGENT_PKG_DIR / "data" / "agent_output"

# Project root for .env resolution
PROJECT_ROOT = AGENT_PKG_DIR.parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()  # Fallback to default behavior

# Magic-value constants
_MIN_PDF_CONTENT_LENGTH = 50
_GSTIN_EXPECTED_LENGTH = 15


@dataclass
class _GCPConfig:
    """Mutable container for GCP configuration (lazy-initialized)."""

    PROJECT_ID: str | None = None
    LOCATION: str = "us-central1"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    GEMINI_PRO_MODEL: str = "gemini-2.5-flash"
    API_CALL_DELAY_SECONDS: float = 1.0
    initialized: bool = False


_gcp_config = _GCPConfig(
    LOCATION=os.getenv("LOCATION", "us-central1"),
    GEMINI_FLASH_MODEL=os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
    GEMINI_PRO_MODEL=os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-flash"),
    API_CALL_DELAY_SECONDS=float(os.getenv("API_CALL_DELAY_SECONDS", "1.0")),
)


def _ensure_gcp_initialized():
    """Lazy-initialize GCP/Vertex AI on first use (not at import time).

    Agent Engine sets env vars after module import, so we must defer."""
    if _gcp_config.initialized:
        return
    _gcp_config.PROJECT_ID = (
        os.getenv("PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        or os.getenv("GCP_PROJECT")
    )
    _gcp_config.LOCATION = os.getenv("LOCATION") or os.getenv(
        "GOOGLE_CLOUD_REGION", "us-central1"
    )
    _gcp_config.GEMINI_FLASH_MODEL = os.getenv(
        "GEMINI_FLASH_MODEL", "gemini-2.5-flash"
    )
    _gcp_config.GEMINI_PRO_MODEL = os.getenv(
        "GEMINI_PRO_MODEL", "gemini-2.5-flash"
    )
    _gcp_config.API_CALL_DELAY_SECONDS = float(
        os.getenv("API_CALL_DELAY_SECONDS", "1.0")
    )
    if not _gcp_config.PROJECT_ID:
        print("Warning: PROJECT_ID not found in environment")
        print("Set it in .env file or export PROJECT_ID=your-gcp-project-id")
    else:
        aiplatform.init(
            project=_gcp_config.PROJECT_ID, location=_gcp_config.LOCATION
        )
    _gcp_config.initialized = True


# Default configuration - can be overridden via config file
DEFAULT_CONFIG = {
    # Organization names for customer verification
    "organization_names": [],  # Empty = skip customer name verification
    # Tax settings
    "default_tax_rate": 0.18,  # 18% (Indian GST standard rate)
    "tax_rates_by_currency": {
        "INR": 0.18,
        "USD": 0.00,
        "EUR": 0.20,
    },
    # Validation settings
    "require_work_authorization": False,
    "waf_exempt_work_types": ["PREVENTATIVE", "CLEANING"],
    "waf_exempt_vendors": [],
    # PO validation
    "require_purchase_order": False,
    "valid_po_prefixes": ["PO", "WO", "PR"],
    # Duplicate detection
    "duplicate_check_enabled": False,
    "duplicate_check_type": "none",  # "none", "file", "database", "api"
    # Tolerances
    "balance_tolerance": 0.02,
    "line_sum_tolerance": 1.00,
    "hours_tolerance": 0.5,
    # Tax ID validation
    "validate_tax_id_checksum": True,
    "tax_id_format": "GSTIN",  # "GSTIN", "ABN", "VAT", "EIN", "NONE"
}

# Module-level mutable containers for config and metrics
_config_store: dict[str, Any] = {"CONFIG": DEFAULT_CONFIG.copy()}

_metrics_store: dict[str, Any] = {
    "METRICS": {
        "llm_calls": 0,
        "total_tokens": {"prompt": 0, "completion": 0},
        "total_cost_usd": 0.0,
        "agent_breakdown": [],
    }
}


def _get_config() -> dict:
    """Return the current CONFIG dict."""
    return _config_store["CONFIG"]


def _get_metrics() -> dict:
    """Return the current METRICS dict."""
    return _metrics_store["METRICS"]


# Pricing per 1M tokens
MODEL_PRICING = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================


class DocumentContent(BaseModel):
    """Identifies document types present in a file"""

    has_invoice: bool = False
    has_work_authorization: bool = False
    invoice_count: int = 0
    reasoning: str = ""


class InvoiceLineItem(BaseModel):
    """Single line item from invoice (GST e-Invoice compatible)"""

    sl_no: str | None = None
    description: str
    is_service: str | None = "N"
    hsn_cd: str | None = ""
    quantity: float | None = None
    free_qty: float | None = 0.0
    unit: str | None = ""
    unit_price: float | None = None
    amount_ex_tax: float | None = None
    discount: float | None = 0.0
    pre_tax_val: float | None = 0.0
    ass_amt: float | None = 0.0
    gst_rt: float | None = 0.0
    cgst_amt: float | None = 0.0
    sgst_amt: float | None = 0.0
    igst_amt: float | None = 0.0
    ces_rt: float | None = 0.0
    ces_amt: float | None = 0.0
    oth_chrg: float | None = 0.0
    tot_item_val: float | None = 0.0
    tax_code: str | None = "TAX"
    tax_amount: float | None = None
    amount_inc_tax: float | None = None


class AddressDetails(BaseModel):
    """Address details for seller/buyer/dispatch/ship"""

    addr1: str | None = ""
    addr2: str | None = ""
    loc: str | None = ""
    pin: int | None = None
    stcd: str | None = ""
    ph: str | None = None
    em: str | None = None


class InvoiceExtraction(BaseModel):
    """Extracted invoice data (GST e-Invoice compatible)"""

    invoice_number: str | None = "UNKNOWN"
    invoice_date: str | None = ""
    invoice_type: str | None = "INV"
    invoice_total_inc_tax: float | None = 0.0
    invoice_total_ex_tax: float | None = 0.0
    tax_amount: float | None = 0.0
    cgst_val: float | None = 0.0
    sgst_val: float | None = 0.0
    igst_val: float | None = 0.0
    ces_val: float | None = 0.0
    discount: float | None = 0.0
    oth_chrg: float | None = 0.0
    rnd_off_amt: float | None = None
    vendor_tax_id: str | None = None
    vendor_name: str | None = "UNKNOWN"
    vendor_trade_name: str | None = ""
    vendor_address: AddressDetails | None = None
    customer_tax_id: str | None = None
    customer_name: str | None = None
    customer_trade_name: str | None = ""
    customer_pos: str | None = ""
    customer_address: AddressDetails | None = None
    ship_to_gstin: str | None = ""
    ship_to_name: str | None = ""
    ship_to_address: AddressDetails | None = None
    irn: str | None = None
    ack_no: str | None = None
    ack_dt: str | None = None
    currency: str | None = "INR"
    supply_type: str | None = ""
    reverse_charge: str | None = "N"
    line_items: list[InvoiceLineItem] | None = []


class WorkAuthorizationExtraction(BaseModel):
    """Extracted work authorization data"""

    reference_number: str | None = None
    site_name: str | None = None
    authorized_hours: float | None = None
    work_description: str | None = None
    technician_name: str | None = None
    date: str | None = None


class WorkTypeClassification(BaseModel):
    """LLM response for work type classification"""

    work_type: Literal["REPAIRS", "PREVENTATIVE", "CLEANING", "EMERGENCY"] = (
        Field(description="The work type category")
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ItemCodeClassification(BaseModel):
    """LLM response for item code classification"""

    item_code: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class VendorNameSimilarity(BaseModel):
    """LLM response for vendor name matching"""

    are_similar: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from file or use defaults.

    Config path is resolved relative to SCRIPT_DIR if not absolute.
    """
    if config_path:
        config_file = Path(config_path)
        # Resolve relative paths relative to script directory
        if not config_file.is_absolute():
            config_file = SCRIPT_DIR / config_file

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                user_config = json.load(f)
                _config_store["CONFIG"] = {**DEFAULT_CONFIG, **user_config}
                print(f"Loaded config from {config_file}")
        else:
            print(f"Warning: Config file not found: {config_file}")
            _config_store["CONFIG"] = DEFAULT_CONFIG.copy()
    else:
        _config_store["CONFIG"] = DEFAULT_CONFIG.copy()

    return _config_store["CONFIG"]


def get_output_folder(case_id: str) -> Path:
    """Get output folder for a case, create if not exists"""
    output_folder = OUTPUT_BASE_DIR / case_id
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def extract_pdf_to_markdown(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber"""
    markdown_lines = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                markdown_lines.append(f"## Page {page_num}\n")
                text = page.extract_text()
                if text:
                    markdown_lines.append(text)
                    markdown_lines.append("\n---\n")
        return "\n".join(markdown_lines)
    except Exception as e:
        print(f"    Error extracting PDF {pdf_path}: {e}")
        return f"[PDF extraction failed: {e}]"


def extract_pdf_with_gemini(pdf_path: Path) -> str:
    """Extract text from PDF using Gemini multimodal"""
    model = GenerativeModel(
        _gcp_config.GEMINI_FLASH_MODEL,
        generation_config=GenerationConfig(temperature=0),
    )

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    pdf_part = Part.from_data(mime_type="application/pdf", data=pdf_data)

    prompt = """Extract ALL text content from this PDF document.
Return the text exactly as it appears, preserving numbers, dates, amounts, and structure.
Do not summarize - extract complete text content."""

    response = model.generate_content([pdf_part, prompt])

    if _gcp_config.API_CALL_DELAY_SECONDS > 0:
        time.sleep(_gcp_config.API_CALL_DELAY_SECONDS)

    return response.text


def extract_pdf_with_fallback(pdf_path: Path) -> tuple[str, str]:
    """Extract PDF with Gemini as default, pdfplumber as fallback"""
    _ensure_gcp_initialized()
    if _gcp_config.PROJECT_ID:
        try:
            content = extract_pdf_with_gemini(pdf_path)
            if content and len(content.strip()) > _MIN_PDF_CONTENT_LENGTH:
                return content, "gemini"
        except Exception as e:
            print(f"    Gemini extraction failed: {e}, using pdfplumber...")

    content = extract_pdf_to_markdown(pdf_path)
    return content, "pdfplumber"


def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code-block fences from text, returning the inner content."""
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    json_lines: list[str] = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```"):
            if in_block:
                break
            in_block = True
            continue
        elif in_block:
            json_lines.append(line)
    if json_lines:
        return "\n".join(json_lines).strip()
    return text


def _extract_json_object(text: str) -> str:
    """Find and return the first top-level JSON object in *text*."""
    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object found")

    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        raise ValueError("Unclosed JSON object")

    return text[start_idx:end_idx]


def clean_json_response(response_text: str) -> str:
    """Clean LLM JSON response - remove markdown, fix trailing commas"""
    text = _strip_markdown_code_block(response_text.strip())
    json_str = _extract_json_object(text)
    # Remove trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    return json_str


def normalize_tax_id(tax_id: str, alphanumeric: bool = False) -> str:
    """Remove whitespace and special characters from tax ID.

    Args:
        tax_id: Raw tax ID string.
        alphanumeric: If True, keep letters too (for GSTIN). Otherwise digits only.
    """
    if alphanumeric:
        return re.sub(r"[^0-9A-Za-z]", "", tax_id or "").upper()
    return re.sub(r"[^0-9]", "", tax_id or "")


def validate_gstin_checksum(gstin: str) -> tuple[bool, str]:
    """Validate Indian GST Identification Number (GSTIN).

    GSTIN format (15 chars): SSPPPPPPPPPPXCZD
      SS   = 2-digit state code (01-37, 97)
      PPPPPPPPPP = 10-char PAN
      X    = entity code (1-9 or A-Z)
      C    = check character (alphanumeric)
      Z/D  = default 'Z' + check digit
    """
    gstin_clean = normalize_tax_id(gstin, alphanumeric=True)

    if len(gstin_clean) != _GSTIN_EXPECTED_LENGTH:
        return (
            False,
            f"Invalid length: {len(gstin_clean)} (must be {_GSTIN_EXPECTED_LENGTH})",
        )

    # State code validation (first 2 chars must be digits)
    if not gstin_clean[0:2].isdigit():
        return False, "First 2 characters must be state code (digits)"

    state_code = int(gstin_clean[0:2])
    valid_state_codes = set(range(1, 38)) | {97}
    if state_code not in valid_state_codes:
        return False, f"Invalid state code: {state_code:02d} (valid: 01-37, 97)"

    # PAN format check (chars 3-12): 5 letters + 4 digits + 1 letter
    pan = gstin_clean[2:12]
    if not (pan[:5].isalpha() and pan[5:9].isdigit() and pan[9].isalpha()):
        return False, f"Invalid PAN format in positions 3-12: {pan}"

    # Entity code (char 13): must be alphanumeric
    if not gstin_clean[12].isalnum():
        return False, "Character 13 (entity code) must be alphanumeric"

    # Character 14 is check character (alphanumeric) — validated via checksum below
    # Character 15: typically 'Z' but some GSTINs use other values

    # Mod-36 checksum validation (Luhn-like for alphanumeric)
    _GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    try:
        factor = 1
        total = 0
        for i in range(len(gstin_clean) - 1):
            code_point = _GSTIN_CHARS.index(gstin_clean[i])
            digit = factor * code_point
            digit = (digit // 36) + (digit % 36)
            total += digit
            factor = 2 if factor == 1 else 1

        remainder = total % 36
        check_char = _GSTIN_CHARS[(36 - remainder) % 36]

        if check_char == gstin_clean[-1]:
            return True, "Valid GSTIN checksum"
        return (
            False,
            f"Invalid check character: '{gstin_clean[-1]}' (expected '{check_char}')",
        )
    except (ValueError, IndexError) as e:
        return False, f"Checksum validation error: {e}"


def validate_tax_id(tax_id: str, format_type: str = "GSTIN") -> tuple[bool, str]:
    """Validate tax ID based on format type"""
    if format_type == "NONE":
        return True, "Validation disabled"

    if format_type == "GSTIN":
        return validate_gstin_checksum(tax_id)

    if format_type == "ABN":
        # Legacy Australian ABN support (11 digits, mod 89)
        abn_clean = normalize_tax_id(tax_id)
        if len(abn_clean) != 11:
            return False, f"Invalid ABN length: {len(abn_clean)} (must be 11)"
        weights = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        checksum = (int(abn_clean[0]) - 1) * weights[0]
        for i in range(1, 11):
            checksum += int(abn_clean[i]) * weights[i]
        if checksum % 89 == 0:
            return True, "Valid ABN checksum"
        return False, f"Invalid ABN checksum (mod 89 = {checksum % 89})"

    # Add other formats as needed (VAT, EIN, etc.)
    return True, "Format validation not implemented"


def call_gemini(
    prompt: str,
    model_name: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> tuple[Any, float]:
    """Call Gemini API with optional structured output"""
    metrics = _get_metrics()
    model_name = model_name or _gcp_config.GEMINI_FLASH_MODEL
    model = GenerativeModel(
        model_name,
        generation_config=GenerationConfig(temperature=0),
    )

    start_time = time.time()
    response = model.generate_content(prompt)
    latency_ms = (time.time() - start_time) * 1000

    # Update metrics
    usage = response.usage_metadata
    metrics["llm_calls"] += 1
    metrics["total_tokens"]["prompt"] += usage.prompt_token_count
    metrics["total_tokens"]["completion"] += usage.candidates_token_count

    pricing = MODEL_PRICING.get(
        model_name.split("/")[-1], MODEL_PRICING["gemini-2.5-flash"]
    )
    cost = (usage.prompt_token_count / 1_000_000) * pricing["input"]
    cost += (usage.candidates_token_count / 1_000_000) * pricing["output"]
    metrics["total_cost_usd"] += cost

    if _gcp_config.API_CALL_DELAY_SECONDS > 0:
        time.sleep(_gcp_config.API_CALL_DELAY_SECONDS)

    if response_schema:
        json_str = clean_json_response(response.text)
        return response_schema.model_validate_json(json_str), latency_ms

    return response.text, latency_ms


def call_gemini_with_pdf(
    pdf_path: Path,
    prompt: str,
    model_name: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> tuple[Any, float]:
    """Call Gemini with PDF as input"""
    metrics = _get_metrics()
    model_name = model_name or _gcp_config.GEMINI_PRO_MODEL
    model = GenerativeModel(
        model_name,
        generation_config=GenerationConfig(temperature=0),
    )

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    pdf_part = Part.from_data(mime_type="application/pdf", data=pdf_data)

    start_time = time.time()
    response = model.generate_content([pdf_part, prompt])
    latency_ms = (time.time() - start_time) * 1000

    # Update metrics
    usage = response.usage_metadata
    metrics["llm_calls"] += 1
    metrics["total_tokens"]["prompt"] += usage.prompt_token_count
    metrics["total_tokens"]["completion"] += usage.candidates_token_count

    pricing = MODEL_PRICING.get(
        model_name.split("/")[-1], MODEL_PRICING["gemini-2.5-flash"]
    )
    cost = (usage.prompt_token_count / 1_000_000) * pricing["input"]
    cost += (usage.candidates_token_count / 1_000_000) * pricing["output"]
    metrics["total_cost_usd"] += cost

    if _gcp_config.API_CALL_DELAY_SECONDS > 0:
        time.sleep(_gcp_config.API_CALL_DELAY_SECONDS)

    if response_schema:
        json_str = clean_json_response(response.text)
        return response_schema.model_validate_json(json_str), latency_ms

    return response.text, latency_ms


def parse_date(date_str: str | None) -> date | None:
    """Parse date string in various formats"""
    if not date_str:
        return None
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"]:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def check_vendor_name_similarity(
    name1: str, name2: str
) -> tuple[bool, str, float]:
    """Use LLM to check if vendor names are semantically equivalent"""
    prompt = f"""Determine if these two vendor names refer to the SAME business entity.

Name 1: "{name1}"
Name 2: "{name2}"

Consider as SIMILAR:
- Trading names vs legal names
- Abbreviations (Pty Ltd = P/L = PTY LTD)
- Case differences
- Minor punctuation differences

Return ONLY this JSON:
{{"are_similar": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        result, _ = call_gemini(
            prompt, _gcp_config.GEMINI_FLASH_MODEL, VendorNameSimilarity
        )
        return result.are_similar, result.reasoning, result.confidence
    except Exception as e:
        return False, f"LLM check failed: {e}", 0.0


# ============================================================================
# BASE AGENT CLASS
# ============================================================================


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version

    def create_output(
        self, data: dict, input_refs: list[str] | None = None
    ) -> dict:
        """Wrap agent output with metadata"""
        return {
            "agent": self.name,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "input_refs": input_refs or [],
            **data,
        }

    def save_artifact(
        self, output_folder: Path, filename: str, data: dict
    ) -> Path:
        """Save artifact to output folder"""
        output_file = output_folder / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  => Saved: {filename}")
        return output_file

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        """Execute the agent logic"""
        pass


# ============================================================================
# AGENT 1: CLASSIFIER
# ============================================================================


class ClassifierAgent(BaseAgent):
    """Document Classification Agent"""

    def __init__(self):
        super().__init__("classifier")

    def run(self, source_folder: Path, output_folder: Path) -> dict:
        print(" [1/9] Classifier Agent: Starting...")

        files_info = {}
        summary = {"invoice_count": 0, "waf_count": 0}
        invoice_sources = []
        waf_sources = []

        for file_path in source_folder.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() == ".pdf":
                content, method = extract_pdf_with_fallback(file_path)

                # Analyze content
                doc_info = self._analyze_document(content, file_path.name)

                if doc_info.has_invoice:
                    invoice_sources.append(
                        {
                            "path": file_path.name,
                            "full_path": str(file_path),
                            "content": content,
                        }
                    )
                    summary["invoice_count"] += doc_info.invoice_count

                if doc_info.has_work_authorization:
                    waf_sources.append(
                        {
                            "path": file_path.name,
                            "full_path": str(file_path),
                            "content": content,
                        }
                    )
                    summary["waf_count"] += 1

                files_info[file_path.name] = {
                    "type": "pdf",
                    "extraction_method": method,
                    "has_invoice": doc_info.has_invoice,
                    "has_waf": doc_info.has_work_authorization,
                }

            elif file_path.suffix.lower() == ".json":
                files_info[file_path.name] = {"type": "json"}

        files_info["invoice_sources"] = invoice_sources
        files_info["waf_sources"] = waf_sources

        output = self.create_output(
            {
                "source_folder": source_folder.name,
                "files": files_info,
                "summary": summary,
            }
        )

        self.save_artifact(output_folder, "01_classification.json", output)
        print(
            f"   Found {summary['invoice_count']} invoice(s), {summary['waf_count']} WAF(s)"
        )

        return output

    def _analyze_document(self, content: str, filename: str) -> DocumentContent:
        """Analyze document content to identify type"""
        prompt = f"""Analyze this document and identify what types are present.

DOCUMENT CONTENT:
{content[:8000]}

INVOICE INDICATORS:
- Invoice number field
- Line items table with amounts
- Total amounts (subtotal, tax, total)
- Vendor business details and tax ID

WORK AUTHORIZATION INDICATORS:
- Work authorization/WAF header
- Site attendance records
- Authorized hours
- Technician signatures

IMPORTANT: If a single PDF contains multiple copies of the SAME invoice
(e.g., "Original for Recipient", "Duplicate for Transporter",
"Triplicate for Supplier" — common in Indian tax invoices), count it as
invoice_count: 1, NOT multiple invoices. Only count truly distinct invoices
with different invoice numbers.

Return ONLY this JSON:
{{
  "has_invoice": true/false,
  "has_work_authorization": true/false,
  "invoice_count": 0-N,
  "reasoning": "brief explanation"
}}"""

        try:
            result, _ = call_gemini(
                prompt, _gcp_config.GEMINI_FLASH_MODEL, DocumentContent
            )
            return result
        except Exception as e:
            print(f"    Warning: Document analysis failed: {e}")
            # Fallback to keyword detection
            content_lower = content.lower()
            return DocumentContent(
                has_invoice="invoice" in content_lower
                and "total" in content_lower,
                has_work_authorization="work authorization" in content_lower
                or "waf" in content_lower,
                invoice_count=1 if "invoice" in content_lower else 0,
                reasoning="Keyword fallback",
            )


# ============================================================================
# AGENT 2: EXTRACTOR
# ============================================================================


class ExtractorAgent(BaseAgent):
    """Invoice/WAF Extraction Agent"""

    def __init__(self):
        super().__init__("extractor")

    def run(
        self, source_folder: Path, output_folder: Path, classification: dict
    ) -> dict:
        print(" [2/9] Extractor Agent: Starting...")

        files_info = classification.get("files", {})
        invoice_sources = files_info.get("invoice_sources", [])
        waf_sources = files_info.get("waf_sources", [])

        # Extract invoice
        invoice_data = None
        extraction_failed = False
        extraction_error = None

        if invoice_sources:
            source = invoice_sources[0]
            try:
                invoice_data = self._extract_invoice(Path(source["full_path"]))

                # Validate tax ID if configured
                if _get_config().get(
                    "validate_tax_id_checksum"
                ) and invoice_data.get("vendor_tax_id"):
                    tax_valid, tax_reason = validate_tax_id(
                        invoice_data["vendor_tax_id"],
                        _get_config().get("tax_id_format", "GSTIN"),
                    )
                    invoice_data["_tax_id_validation"] = {
                        "valid": tax_valid,
                        "reason": tax_reason,
                    }

                print(
                    f"   Extracted invoice: {invoice_data.get('invoice_number', 'N/A')}"
                )
            except Exception as e:
                extraction_failed = True
                extraction_error = str(e)
                invoice_data = self._empty_invoice()
        else:
            extraction_failed = True
            extraction_error = "No invoice found"
            invoice_data = self._empty_invoice()

        # Extract WAF if present
        waf_data = None
        if waf_sources:
            try:
                waf_data = self._extract_waf(Path(waf_sources[0]["full_path"]))
                print(
                    f"   Extracted WAF: {waf_data.get('authorized_hours', 0)} hours"
                )
            except Exception as e:
                print(f"   WAF extraction failed: {e}")

        output = self.create_output(
            {
                "invoice": invoice_data,
                "work_authorization": waf_data,
                "extraction_failed": extraction_failed,
                "extraction_error": extraction_error,
                "invoice_count": classification.get("summary", {}).get(
                    "invoice_count", 0
                ),
                "waf_count": classification.get("summary", {}).get(
                    "waf_count", 0
                ),
            },
            input_refs=["01_classification.json"],
        )

        self.save_artifact(output_folder, "02_extraction.json", output)
        return output

    def _extract_invoice(self, pdf_path: Path) -> dict:
        """Extract structured invoice data from PDF (GST e-Invoice compatible)"""
        prompt = """Extract structured data from this Indian GST invoice PDF.
If the PDF has multiple copies of the same invoice (Original, Duplicate,
Triplicate), extract from the FIRST copy only.

Return ONLY valid JSON with these fields:
{
  "invoice_number": "string (Invoice No / Bill No)",
  "invoice_date": "DD/MM/YYYY",
  "invoice_type": "INV or CRN or DBN",
  "invoice_total_inc_tax": decimal,
  "invoice_total_ex_tax": decimal (taxable value / assessable value)",
  "tax_amount": decimal (total tax),
  "cgst_val": decimal (total CGST amount),
  "sgst_val": decimal (total SGST amount),
  "igst_val": decimal (total IGST amount),
  "ces_val": decimal (total Cess amount or 0),
  "discount": decimal (total discount or 0),
  "oth_chrg": decimal (other charges or 0),
  "rnd_off_amt": decimal or null (rounding off amount),
  "vendor_tax_id": "string (seller GSTIN - exactly 15 chars)",
  "vendor_name": "string (seller legal name)",
  "vendor_trade_name": "string (seller trade name if different)",
  "vendor_address": {
    "addr1": "string", "addr2": "string", "loc": "string",
    "pin": integer or null, "stcd": "string (2-digit state code)",
    "ph": "string or null", "em": "string or null"
  },
  "customer_tax_id": "string (buyer GSTIN - exactly 15 chars)",
  "customer_name": "string (buyer legal name from Bill To)",
  "customer_trade_name": "string (buyer trade name if different)",
  "customer_pos": "string (Place of Supply state code)",
  "customer_address": {
    "addr1": "string", "addr2": "string", "loc": "string",
    "pin": integer or null, "stcd": "string (2-digit state code)",
    "ph": "string or null", "em": "string or null"
  },
  "ship_to_gstin": "string (Consignee GSTIN if different from buyer)",
  "ship_to_name": "string (Consignee name if different)",
  "ship_to_address": {
    "addr1": "string", "addr2": "string", "loc": "string",
    "pin": integer or null, "stcd": "string"
  },
  "irn": "string or null (Invoice Reference Number - 64 hex chars, strip hyphens)",
  "ack_no": "string or null (Acknowledgement Number)",
  "ack_dt": "string or null (Acknowledgement Date DD/MM/YYYY)",
  "currency": "INR",
  "supply_type": "string (B2B, B2C, SEZWP, SEZWOP, EXPWP, EXPWOP, DEXP)",
  "reverse_charge": "Y or N",
  "line_items": [
    {
      "sl_no": "string (serial number)",
      "description": "string (product/service description)",
      "is_service": "Y or N",
      "hsn_cd": "string (HSN/SAC code)",
      "quantity": decimal or null,
      "free_qty": decimal or 0,
      "unit": "string (UOM - EA, KG, NOS, etc.)",
      "unit_price": decimal or null,
      "amount_ex_tax": decimal (line total before tax),
      "discount": decimal or 0,
      "pre_tax_val": decimal or 0,
      "ass_amt": decimal (assessable amount for this line),
      "gst_rt": decimal (combined GST rate e.g. 18 for 9+9),
      "cgst_amt": decimal (CGST amount - use printed value only),
      "sgst_amt": decimal (SGST amount - use printed value only),
      "igst_amt": decimal (IGST amount - use printed value only),
      "ces_rt": decimal or 0,
      "ces_amt": decimal or 0,
      "oth_chrg": decimal or 0,
      "tot_item_val": decimal (final line total including tax),
      "tax_code": "CGST/SGST/IGST/GST/TAX/NA",
      "tax_amount": decimal or null (total tax for this line)
    }
  ]
}

IMPORTANT RULES:
- Use ONLY printed values. Do NOT recalculate totals.
- IRN: strip hyphens/spaces, must be exactly 64 hex chars or null.
- GSTIN: must be exactly 15 alphanumeric characters.
- For tax split: use CGST/SGST if intra-state, IGST if inter-state.
- gst_rt: if only CGST 9% and SGST 9% printed, set gst_rt to 18.
- Use 0 for missing numeric fields, "" for missing text, null for IRN/AckNo if absent.
- Extract ALL line items including transport charges, freight, other services.
- DISCOUNT HANDLING: If a discount is printed on a line item:
  - Set "discount" to the printed discount amount.
  - Set "ass_amt" to the NET assessable amount AFTER discount (not the gross amount).
  - Set "amount_ex_tax" to the same net value after discount.
  - Tax (CGST/SGST) should be calculated on the net (post-discount) amount.
  - Set "pre_tax_val" to the net assessable amount (same as ass_amt).
- PHONE NUMBERS: Extract seller/buyer phone numbers into the address ph field.
- SUPPLY TYPE: Set supply_type to "B2B" if both seller and buyer have GSTIN."""

        result, _ = call_gemini_with_pdf(
            pdf_path, prompt, _gcp_config.GEMINI_PRO_MODEL, InvoiceExtraction
        )
        return result.model_dump()

    def _extract_waf(self, pdf_path: Path) -> dict:
        """Extract work authorization data from PDF"""
        prompt = """Extract work authorization data from this PDF.

Return ONLY valid JSON:
{
  "reference_number": "string or null",
  "site_name": "string or null",
  "authorized_hours": decimal or null,
  "work_description": "string or null",
  "technician_name": "string or null",
  "date": "YYYY-MM-DD or null"
}"""

        result, _ = call_gemini_with_pdf(
            pdf_path,
            prompt,
            _gcp_config.GEMINI_PRO_MODEL,
            WorkAuthorizationExtraction,
        )
        return result.model_dump()

    def _empty_invoice(self) -> dict:
        """Return empty invoice structure (GST e-Invoice compatible)"""
        return {
            "invoice_number": "EXTRACTION_FAILED",
            "invoice_date": "",
            "invoice_type": "INV",
            "invoice_total_inc_tax": 0.0,
            "invoice_total_ex_tax": 0.0,
            "tax_amount": 0.0,
            "cgst_val": 0.0,
            "sgst_val": 0.0,
            "igst_val": 0.0,
            "ces_val": 0.0,
            "discount": 0.0,
            "oth_chrg": 0.0,
            "rnd_off_amt": None,
            "vendor_tax_id": "",
            "vendor_name": "UNKNOWN",
            "vendor_trade_name": "",
            "vendor_address": None,
            "customer_tax_id": None,
            "customer_name": None,
            "customer_trade_name": "",
            "customer_pos": "",
            "customer_address": None,
            "ship_to_gstin": "",
            "ship_to_name": "",
            "ship_to_address": None,
            "irn": None,
            "ack_no": None,
            "ack_dt": None,
            "currency": "INR",
            "supply_type": "",
            "reverse_charge": "N",
            "line_items": [],
        }


# ============================================================================
# AGENT 3: PHASE 1 VALIDATOR
# ============================================================================


class Phase1ValidatorAgent(BaseAgent):
    """Phase 1: Initial Intake Validation"""

    def __init__(self):
        super().__init__("phase1_validator")

    def run(self, output_folder: Path, extraction: dict) -> dict:
        print(" [3/9] Phase 1 Validator: Starting...")

        invoice = extraction.get("invoice", {})
        validations = []

        # Step 1.1: Extraction Success
        if extraction.get("extraction_failed"):
            validations.append(
                {
                    "step": "1.1",
                    "rule": "Invoice extraction must succeed",
                    "passed": False,
                    "evidence": extraction.get(
                        "extraction_error", "Extraction failed"
                    ),
                    "rejection_template": "Document is not a valid Tax Invoice",
                }
            )
        else:
            validations.append(
                {
                    "step": "1.1",
                    "rule": "Invoice extraction must succeed",
                    "passed": True,
                    "evidence": f"Extracted invoice {invoice.get('invoice_number')}",
                }
            )

        # Step 1.2: Customer Verification
        customer_name = (invoice.get("customer_name") or "").lower()
        org_names = [
            n.lower() for n in _get_config().get("organization_names", [])
        ]
        customer_match = (
            any(org in customer_name for org in org_names)
            if org_names
            else True
        )

        validations.append(
            {
                "step": "1.2",
                "rule": "Invoice addressed to organization",
                "passed": customer_match,
                "evidence": f"Customer: {invoice.get('customer_name')}",
                "rejection_template": None
                if customer_match
                else "Invoice addressed to different company",
            }
        )

        # Step 1.3: Tax Compliance
        has_tax_id = bool(invoice.get("vendor_tax_id"))
        has_date = bool(invoice.get("invoice_date"))
        has_vendor = bool(
            invoice.get("vendor_name")
            and invoice.get("vendor_name") != "UNKNOWN"
        )
        tax_compliant = has_tax_id and has_date and has_vendor

        validations.append(
            {
                "step": "1.3",
                "rule": "Tax compliance (tax ID, date, vendor)",
                "passed": tax_compliant,
                "evidence": f"Tax ID: {has_tax_id}, Date: {has_date}, Vendor: {has_vendor}",
                "rejection_template": None
                if tax_compliant
                else "Document is not a Tax Invoice",
            }
        )

        # Step 1.4: Work Authorization Check
        if _get_config().get("require_work_authorization"):
            work_type = self._determine_work_type(invoice)
            exempt_types = _get_config().get("waf_exempt_work_types", [])
            waf_required = work_type not in exempt_types
            has_waf = extraction.get("waf_count", 0) > 0

            if waf_required:
                validations.append(
                    {
                        "step": "1.4",
                        "rule": "Work authorization required",
                        "passed": has_waf,
                        "evidence": f"WAF required for {work_type}, WAF present: {has_waf}",
                        "rejection_template": None
                        if has_waf
                        else "Missing Work Authorization Form",
                    }
                )

        # Step 1.5: Single Invoice Check
        invoice_count = extraction.get("invoice_count", 1)
        validations.append(
            {
                "step": "1.5",
                "rule": "Single invoice per submission",
                "passed": invoice_count <= 1,
                "evidence": f"Found {invoice_count} invoice(s)",
                "rejection_template": None
                if invoice_count <= 1
                else "Multiple invoices in submission",
            }
        )

        # Determine decision
        failed = [v for v in validations if not v.get("passed")]
        decision = "REJECT" if failed else "CONTINUE"

        output = self.create_output(
            {
                "phase": 1,
                "validations": validations,
                "decision": decision,
                "rejection_template": failed[0].get("rejection_template")
                if failed
                else None,
                "rejection_reason": failed[0].get("evidence")
                if failed
                else None,
            },
            input_refs=["02_extraction.json"],
        )

        self.save_artifact(output_folder, "03_phase1_validation.json", output)
        print(f"   Phase 1: {decision}")
        return output

    def _determine_work_type(self, invoice: dict) -> str:
        """Determine work type from invoice content"""
        descriptions = " ".join(
            [
                (line.get("description") or "").lower()
                for line in (invoice.get("line_items") or [])
            ]
        )

        if any(
            kw in descriptions
            for kw in ["preventative", "pm", "scheduled", "maintenance"]
        ):
            return "PREVENTATIVE"
        if any(
            kw in descriptions for kw in ["cleaning", "clean", "janitorial"]
        ):
            return "CLEANING"
        if any(
            kw in descriptions for kw in ["emergency", "urgent", "after hours"]
        ):
            return "EMERGENCY"
        return "REPAIRS"


# ============================================================================
# AGENT 4: PHASE 2 VALIDATOR
# ============================================================================


class Phase2ValidatorAgent(BaseAgent):
    """Phase 2: Content Validation"""

    def __init__(self):
        super().__init__("phase2_validator")

    def run(self, output_folder: Path, extraction: dict, phase1: dict) -> dict:
        print(" [4/9] Phase 2 Validator: Starting...")

        invoice = extraction.get("invoice", {})
        validations = []

        # Step 2.1: Line Items Present
        line_count = len(invoice.get("line_items") or [])
        validations.append(
            {
                "step": "2.1",
                "rule": "Invoice has line items",
                "passed": line_count > 0,
                "evidence": f"Found {line_count} line item(s)",
                "rejection_template": None
                if line_count > 0
                else "Invoice charges not itemized",
            }
        )

        # Step 2.2: PO Validation (if required)
        # Note: This would need PO data from preprocessing - simplified here
        if _get_config().get("require_purchase_order"):
            validations.append(
                {
                    "step": "2.2",
                    "rule": "Valid purchase order",
                    "passed": True,  # Would check PO in real implementation
                    "evidence": "PO validation skipped (no preprocessing data)",
                }
            )

        # Determine decision
        failed = [v for v in validations if not v.get("passed")]
        decision = "REJECT" if failed else "CONTINUE"

        output = self.create_output(
            {
                "phase": 2,
                "validations": validations,
                "decision": decision,
                "rejection_template": failed[0].get("rejection_template")
                if failed
                else None,
            },
            input_refs=["02_extraction.json", "03_phase1_validation.json"],
        )

        self.save_artifact(output_folder, "04_phase2_validation.json", output)
        print(f"   Phase 2: {decision}")
        return output


# ============================================================================
# AGENT 5: PHASE 3 VALIDATOR (External)
# ============================================================================


class Phase3ValidatorAgent(BaseAgent):
    """Phase 3: External Validation (duplicates, vendor matching)"""

    def __init__(self):
        super().__init__("phase3_validator")

    def run(self, output_folder: Path, extraction: dict, phase2: dict) -> dict:
        print(" [5/9] Phase 3 Validator: Starting...")

        invoice = extraction.get("invoice", {})
        validations = []

        # Step 3.1: Duplicate Detection
        if _get_config().get("duplicate_check_enabled"):
            # Placeholder - would integrate with actual duplicate check
            validations.append(
                {
                    "step": "3.1",
                    "rule": "Not a duplicate invoice",
                    "passed": True,
                    "evidence": "Duplicate check not implemented",
                }
            )

        # Step 3.2: Tax ID Validation
        tax_validation = invoice.get("_tax_id_validation", {})
        if tax_validation:
            validations.append(
                {
                    "step": "3.2",
                    "rule": "Valid tax ID checksum",
                    "passed": tax_validation.get("valid", True),
                    "evidence": tax_validation.get("reason", "Not validated"),
                    "rejection_template": None
                    if tax_validation.get("valid")
                    else "Vendor tax ID invalid",
                }
            )

        # Step 3.3: Future Date Check
        invoice_date = parse_date(invoice.get("invoice_date"))
        today = date.today()

        if invoice_date:
            is_future = invoice_date > today
            validations.append(
                {
                    "step": "3.3",
                    "rule": "Invoice not future dated",
                    "passed": not is_future,
                    "evidence": f"Invoice date: {invoice.get('invoice_date')}",
                    "rejection_template": None
                    if not is_future
                    else "Invoice is future dated",
                }
            )

        # Determine decision
        failed = [v for v in validations if not v.get("passed")]
        decision = "REJECT" if failed else "CONTINUE"

        output = self.create_output(
            {
                "phase": 3,
                "validations": validations,
                "decision": decision,
                "rejection_template": failed[0].get("rejection_template")
                if failed
                else None,
            },
            input_refs=["02_extraction.json", "04_phase2_validation.json"],
        )

        self.save_artifact(output_folder, "05_phase3_validation.json", output)
        print(f"   Phase 3: {decision}")
        return output


# ============================================================================
# AGENT 6: PHASE 4 VALIDATOR (Calculations)
# ============================================================================


class Phase4ValidatorAgent(BaseAgent):
    """Phase 4: Calculation Validation"""

    def __init__(self):
        super().__init__("phase4_validator")

    def run(self, output_folder: Path, extraction: dict, phase3: dict) -> dict:
        print(" [6/9] Phase 4 Validator: Starting...")

        invoice = extraction.get("invoice", {})
        waf = extraction.get("work_authorization") or {}
        validations = []

        # Step 4.1: Total Verification
        total_inc = invoice.get("invoice_total_inc_tax", 0)
        total_ex = invoice.get("invoice_total_ex_tax", 0)
        tax = invoice.get("tax_amount", 0)

        expected_total = total_ex + tax
        balance = abs(total_inc - expected_total)
        tolerance = _get_config().get("balance_tolerance", 0.02)

        validations.append(
            {
                "step": "4.1",
                "rule": "Total = Subtotal + Tax",
                "passed": balance <= tolerance,
                "evidence": f"Total: {total_inc}, Expected: {expected_total}, Diff: {balance:.2f}",
            }
        )

        # Step 4.2: Line Sum Validation
        line_total = sum(
            (item.get("amount_ex_tax") or 0)
            for item in (invoice.get("line_items") or [])
        )
        line_diff = abs(line_total - total_ex)
        line_tolerance = _get_config().get("line_sum_tolerance", 1.00)

        validations.append(
            {
                "step": "4.2",
                "rule": "Sum of lines = Subtotal",
                "passed": line_diff <= line_tolerance,
                "evidence": f"Line sum: {line_total:.2f}, Subtotal: {total_ex:.2f}, Diff: {line_diff:.2f}",
                "rejection_template": None
                if line_diff <= line_tolerance
                else "Invoice amounts do not reconcile",
            }
        )

        # Step 4.3: WAF Hours Check
        # Determine work type to check WAF exemption
        work_type = self._determine_work_type(invoice)
        exempt_types = [
            t.upper() for t in _get_config().get("waf_exempt_work_types", [])
        ]
        waf_required = work_type not in exempt_types

        invoice_hours = self._calculate_labour_hours(invoice)
        waf_hours = waf.get("authorized_hours", 0) if waf else 0
        hours_tolerance = _get_config().get("hours_tolerance", 0.5)

        if waf and waf_hours and waf_hours > 0:
            # WAF present with authorized hours — check limits
            hours_valid = invoice_hours <= waf_hours + hours_tolerance
            validations.append(
                {
                    "step": "4.3",
                    "rule": "Labour hours within authorization",
                    "passed": hours_valid,
                    "evidence": f"Invoice: {invoice_hours}h, Authorized: {waf_hours}h, Work type: {work_type}",
                    "rejection_template": None
                    if hours_valid
                    else "Invoice does not match work authorization",
                }
            )
        elif waf_required and invoice_hours > 0 and (not waf or not waf_hours):
            # Non-exempt work type has labour hours but no WAF authorization
            validations.append(
                {
                    "step": "4.3",
                    "rule": "Labour hours within authorization",
                    "passed": False,
                    "evidence": (
                        f"Invoice: {invoice_hours}h, Work type: {work_type} (non-exempt), "
                        f"WAF authorized hours: {waf_hours} — no authorization for billed labour"
                    ),
                    "rejection_template": "Invoice does not match work authorization",
                }
            )

        # Determine decision
        failed = [v for v in validations if not v.get("passed")]
        decision = "REJECT" if failed else "ACCEPT"

        output = self.create_output(
            {
                "phase": 4,
                "validations": validations,
                "decision": decision,
                "rejection_template": failed[0].get("rejection_template")
                if failed
                else None,
            },
            input_refs=["02_extraction.json", "05_phase3_validation.json"],
        )

        self.save_artifact(output_folder, "06_phase4_validation.json", output)
        print(f"   Phase 4: {decision}")
        return output

    def _determine_work_type(self, invoice: dict) -> str:
        """Determine work type from invoice line item descriptions."""
        descriptions = " ".join(
            [
                (line.get("description") or "").lower()
                for line in (invoice.get("line_items") or [])
            ]
        )

        if any(
            kw in descriptions
            for kw in ["preventative", "pm", "scheduled", "maintenance"]
        ):
            return "PREVENTATIVE"
        if any(
            kw in descriptions for kw in ["cleaning", "clean", "janitorial"]
        ):
            return "CLEANING"
        if any(
            kw in descriptions for kw in ["emergency", "urgent", "after hours"]
        ):
            return "EMERGENCY"
        return "REPAIRS"

    def _calculate_labour_hours(self, invoice: dict) -> float:
        """Calculate total labour hours from invoice"""
        total = 0.0
        labour_keywords = ["labour", "labor", "technician", "hours"]

        for line in invoice.get("line_items") or []:
            desc = (line.get("description") or "").lower()
            if any(kw in desc for kw in labour_keywords):
                qty = line.get("quantity") or 0
                if qty > 0:
                    total += qty
        return total


# ============================================================================
# AGENT 7: TRANSFORMER
# ============================================================================


class TransformerAgent(BaseAgent):
    """Line Item Transformation Agent"""

    VALID_ITEM_CODES: ClassVar[list[str]] = [
        "LABOUR",
        "LABOUR_AH",
        "PARTS",
        "FREIGHT",
        "TRAVEL",
        "CALLOUT",
        "HIRE",
        "CLEANING",
        "OTHER",
    ]

    # Tax code normalization map — extracted values are mapped to
    # the canonical codes expected by downstream systems and eval.
    TAX_CODE_MAP: ClassVar[dict[str, str]] = {
        "GST": "TAX",
        "gst": "TAX",
        "Gst": "TAX",
        "VAT": "TAX",
        "vat": "TAX",
    }

    def __init__(self):
        super().__init__("transformer")

    def run(self, output_folder: Path, extraction: dict, phase4: dict) -> dict:
        print(" [7/9] Transformer Agent: Starting...")

        invoice = extraction.get("invoice", {})
        currency = invoice.get("currency", "INR")
        tax_rate = (
            _get_config().get("tax_rates_by_currency", {}).get(currency, 0.18)
        )

        line_items = invoice.get("line_items") or []

        # Classify all line items in a single LLM call for efficiency
        descriptions = [line.get("description", "") for line in line_items]
        item_codes = self._classify_items_llm(descriptions)

        mapped_items = []
        for idx, line in enumerate(line_items):
            item_code = item_codes[idx] if idx < len(item_codes) else "OTHER"
            amount = line.get("amount_ex_tax") or 0
            tax = line.get("tax_amount") or (amount * tax_rate)

            # Normalize tax code (e.g. GST -> TAX)
            raw_tax_code = line.get("tax_code", "TAX")
            tax_code = self.TAX_CODE_MAP.get(raw_tax_code, raw_tax_code)

            mapped_items.append(
                {
                    "line_number": idx + 1,
                    "item_code": item_code,
                    "description": line.get("description", ""),
                    "quantity": f"{line.get('quantity') or 1:.2f}",
                    "unit_cost": f"{line.get('unit_price') or amount:,.2f}",
                    "line_cost": f"{amount:,.2f}",
                    "tax": f"{tax:.2f}",
                    "tax_code": tax_code,
                }
            )

        # Calculate totals
        totals = {
            "line_cost_total": sum(
                float(i["line_cost"].replace(",", "")) for i in mapped_items
            ),
            "tax_total": sum(float(i["tax"]) for i in mapped_items),
        }
        totals["grand_total"] = totals["line_cost_total"] + totals["tax_total"]

        output = self.create_output(
            {
                "currency": currency,
                "tax_rate": tax_rate,
                "line_items_mapped": mapped_items,
                "totals": totals,
            },
            input_refs=["02_extraction.json", "06_phase4_validation.json"],
        )

        self.save_artifact(output_folder, "07_transformation.json", output)
        print(f"   Mapped {len(mapped_items)} line items")
        return output

    def _classify_items_llm(self, descriptions: list[str]) -> list[str]:
        """Classify all line item descriptions using a single LLM call.

        Uses Gemini to semantically classify each line item description into
        one of the valid item codes, rather than relying on brittle keyword matching.
        Falls back to 'OTHER' on failure.
        """
        if not descriptions:
            return []

        items_block = "\n".join(
            f'  {i + 1}. "{desc}"' for i, desc in enumerate(descriptions)
        )

        prompt = f"""Classify each invoice line item description into exactly one item code.

VALID ITEM CODES:
- LABOUR: Work performed by technicians, engineers, tradespeople. Includes service calls,
  inspections, diagnostics, calibration, testing, system checks, standard-hours work.
- LABOUR_AH: After-hours or overtime labour only.
- PARTS: Physical materials, components, equipment, replacement parts, filters, coils,
  fittings, valves, units, supplies, consumables — any tangible item purchased/installed.
- FREIGHT: Delivery, shipping, freight charges.
- TRAVEL: Travel time, mileage, kilometre charges, travel allowances.
- CALLOUT: Call-out fees, attendance fees, minimum charges for site visits.
- HIRE: Equipment hire or rental charges.
- CLEANING: Cleaning services, janitorial work.
- OTHER: Only if the description truly does not fit any of the above categories.

LINE ITEMS TO CLASSIFY:
{items_block}

IMPORTANT RULES:
- Physical items (filters, coils, fittings, parts, materials, equipment, units) are PARTS.
- Service work (calibration, diagnostics, testing, inspection, repair labour) is LABOUR.
- Prefer a specific code over OTHER. Only use OTHER as a last resort.

Return ONLY a JSON object in this exact format:
{{"classifications": [{{"item_number": 1, "item_code": "CODE", "reasoning": "brief reason"}}, ...]}}"""

        try:
            result, _ = call_gemini(prompt, _gcp_config.GEMINI_FLASH_MODEL)
            json_str = clean_json_response(result)
            parsed = json.loads(json_str)
            classifications = parsed.get("classifications", [])

            # Build result list, validating each code
            codes = []
            for i, _desc in enumerate(descriptions):
                item_num = i + 1
                match = next(
                    (
                        c
                        for c in classifications
                        if c.get("item_number") == item_num
                    ),
                    None,
                )
                if match and match.get("item_code") in self.VALID_ITEM_CODES:
                    codes.append(match["item_code"])
                else:
                    codes.append("OTHER")
            return codes

        except Exception as e:
            print(
                f"    Warning: LLM item classification failed ({e}), using fallback"
            )
            return self._classify_items_keyword_fallback(descriptions)

    def _classify_items_keyword_fallback(
        self, descriptions: list[str]
    ) -> list[str]:
        """Keyword-based fallback for item classification when LLM is unavailable."""
        KEYWORD_MAP = {
            "LABOUR_AH": ["after hours", "overtime", "a/h", "after-hours"],
            "LABOUR": [
                "labour",
                "labor",
                "technician",
                "installation",
                "service",
                "calibration",
                "diagnostics",
                "inspection",
                "repair",
            ],
            "PARTS": [
                "parts",
                "material",
                "component",
                "supply",
                "filter",
                "coil",
                "fitting",
                "valve",
                "unit",
                "equipment",
                "replacement",
            ],
            "FREIGHT": ["freight", "delivery", "shipping"],
            "TRAVEL": ["travel", "mileage", "kilometre", "kilometer"],
            "CALLOUT": ["call out", "callout", "attendance"],
            "HIRE": ["hire", "rental"],
            "CLEANING": ["cleaning", "clean"],
        }

        codes = []
        for desc in descriptions:
            desc_lower = desc.lower()
            matched = "OTHER"
            for code, keywords in KEYWORD_MAP.items():
                if any(kw in desc_lower for kw in keywords):
                    matched = code
                    break
            codes.append(matched)
        return codes


# ============================================================================
# GST e-INVOICE JSON BUILDER
# ============================================================================


def _addr_dict(addr: dict | None) -> dict:
    """Convert address details to e-Invoice format."""
    if not addr:
        return {"Addr1": "", "Addr2": "", "Loc": "", "Pin": None, "Stcd": ""}
    return {
        "Addr1": addr.get("addr1", "") or "",
        "Addr2": addr.get("addr2", "") or "",
        "Loc": addr.get("loc", "") or "",
        "Pin": addr.get("pin"),
        "Stcd": addr.get("stcd", "") or "",
    }


def build_einvoice_json(extraction: dict) -> dict:
    """Transform enriched extraction into GST e-Invoice JSON schema.

    This produces the GST e-Invoice JSON structure.
    """
    inv = extraction.get("invoice", {})
    v_addr = _addr_dict(inv.get("vendor_address"))
    c_addr = _addr_dict(inv.get("customer_address"))
    s_addr = _addr_dict(inv.get("ship_to_address"))

    # --- Deterministic field derivations ---
    seller_gstin = inv.get("vendor_tax_id", "") or ""
    buyer_gstin = inv.get("customer_tax_id", "") or ""
    seller_name = inv.get("vendor_name", "") or ""
    buyer_name = inv.get("customer_name", "") or ""
    seller_trd = inv.get("vendor_trade_name", "") or ""
    buyer_trd = inv.get("customer_trade_name", "") or ""

    # SupTyp: derive from GSTIN presence if LLM didn't extract it
    sup_typ = inv.get("supply_type", "") or ""
    if not sup_typ:
        if seller_gstin and buyer_gstin:
            sup_typ = "B2B"
        elif seller_gstin and not buyer_gstin:
            sup_typ = "B2C"

    # TrdNm: default to LglNm when empty
    if not seller_trd:
        seller_trd = seller_name
    if not buyer_trd:
        buyer_trd = buyer_name

    # Pos (Place of Supply): derive from buyer state code if empty
    cust_pos = inv.get("customer_pos", "") or ""
    if not cust_pos:
        cust_pos = c_addr.get("Stcd", "") or ""
    # Fallback: derive from buyer GSTIN first 2 digits
    if not cust_pos and len(buyer_gstin) >= 2:
        cust_pos = buyer_gstin[:2]

    # Build ItemList
    item_list = []
    for item in inv.get("line_items") or []:
        item_list.append({
            "SlNo": str(item.get("sl_no") or (len(item_list) + 1)),
            "PrdDesc": item.get("description", ""),
            "IsServc": item.get("is_service", "N") or "N",
            "HsnCd": item.get("hsn_cd", "") or "",
            "Barcde": None,
            "Qty": item.get("quantity") or 0,
            "FreeQty": item.get("free_qty") or 0,
            "Unit": item.get("unit", "") or "",
            "UnitPrice": item.get("unit_price") or 0,
            "TotAmt": item.get("amount_ex_tax") or item.get("ass_amt") or 0,
            "Discount": item.get("discount") or 0,
            "PreTaxVal": item.get("pre_tax_val") or 0,
            "AssAmt": item.get("ass_amt") or item.get("amount_ex_tax") or 0,
            "GstRt": item.get("gst_rt") or 0,
            "IgstAmt": item.get("igst_amt") or 0,
            "CgstAmt": item.get("cgst_amt") or 0,
            "SgstAmt": item.get("sgst_amt") or 0,
            "CesRt": item.get("ces_rt") or 0,
            "CesAmt": item.get("ces_amt") or 0,
            "CesNonAdvlAmt": 0,
            "StateCesRt": 0,
            "StateCesAmt": 0,
            "StateCesNonAdvlAmt": 0,
            "OthChrg": item.get("oth_chrg") or 0,
            "TotItemVal": item.get("tot_item_val") or item.get("amount_inc_tax") or 0,
            "OrdLineRef": None,
            "OrgCntry": None,
            "PrdSlNo": None,
            "BchDtls": {"Nm": None, "Expdt": None, "Wrdt": None},
            "AttribDtls": None,
        })

    return {
        "Version": "1.1",
        "Irn": inv.get("irn"),
        "AckNo": inv.get("ack_no"),
        "AckDt": inv.get("ack_dt"),
        "TranDtls": {
            "TaxSch": "GST",
            "SupTyp": sup_typ,
            "RegRev": inv.get("reverse_charge", "N") or "N",
            "EcmGstin": None,
            "IgstOnIntra": "N",
        },
        "DocDtls": {
            "Typ": inv.get("invoice_type", "INV") or "INV",
            "No": inv.get("invoice_number", "") or "",
            "Dt": inv.get("invoice_date", "") or "",
        },
        "SellerDtls": {
            "Gstin": seller_gstin,
            "LglNm": seller_name,
            "TrdNm": seller_trd,
            **v_addr,
            "Ph": (inv.get("vendor_address") or {}).get("ph"),
            "Em": (inv.get("vendor_address") or {}).get("em"),
        },
        "BuyerDtls": {
            "Gstin": buyer_gstin,
            "LglNm": buyer_name,
            "TrdNm": buyer_trd,
            "Pos": cust_pos,
            **c_addr,
            "Ph": (inv.get("customer_address") or {}).get("ph"),
            "Em": (inv.get("customer_address") or {}).get("em"),
        },
        "DispDtls": {
            "Nm": "", "Addr1": "", "Addr2": "", "Loc": "",
            "Pin": None, "Stcd": "",
        },
        "ShipDtls": {
            "Gstin": inv.get("ship_to_gstin", "") or "",
            "LglNm": inv.get("ship_to_name", "") or "",
            "TrdNm": "",
            **s_addr,
        },
        "ItemList": item_list,
        "ValDtls": {
            "AssVal": inv.get("invoice_total_ex_tax") or 0,
            "CgstVal": inv.get("cgst_val") or 0,
            "SgstVal": inv.get("sgst_val") or 0,
            "IgstVal": inv.get("igst_val") or 0,
            "CesVal": inv.get("ces_val") or 0,
            "StCesVal": 0,
            "Discount": inv.get("discount") or 0,
            "OthChrg": inv.get("oth_chrg") or 0,
            "RndOffAmt": inv.get("rnd_off_amt"),
            "TotInvVal": inv.get("invoice_total_inc_tax") or 0,
            "TotInvValFc": 0,
        },
        "PayDtls": None,
        "RefDtls": None,
        "AddlDocDtls": None,
        "ExpDtls": None,
        "EwbDtls": None,
    }


# ============================================================================
# POST-EXTRACTION VALIDATOR & ENRICHER
# ============================================================================

_PHONE_RE = re.compile(
    r"(?:\+91[-\s]?)?(?:\d{2,5}[-\s/]?)?\d{6,10}(?:[-/]\d{6,10})*"
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")


def validate_and_enrich_einvoice(
    einvoice: dict,
    raw_pdf_text: str | None = None,
    buyer_gstin_list: list[str] | None = None,
) -> dict:
    """Post-extraction validation and enrichment of e-Invoice JSON.

    Phase 1: Arithmetic validation on items vs totals
    Phase 2: Regex extraction for phone/email from raw PDF text
    Phase 3: Buyer GSTIN validation against known list
    Plus: ShipDtls and PreTaxVal deterministic fixes
    """
    items = einvoice.get("ItemList") or []
    val = einvoice.get("ValDtls") or {}

    # ----- Phase 1: Arithmetic Validator -----
    declared_ass = val.get("AssVal") or 0
    if items and declared_ass > 0:
        item_ass_sum = sum(it.get("AssAmt") or 0 for it in items)
        tolerance = max(declared_ass * 0.02, 1.0)

        # Fix individual items if they look like gross (pre-discount) values
        if abs(item_ass_sum - declared_ass) > tolerance:
            for it in items:
                ass = it.get("AssAmt") or 0
                disc = it.get("Discount") or 0
                tot_amt = it.get("TotAmt") or 0

                # If TotAmt = AssAmt and discount exists, AssAmt is gross
                if disc > 0 and abs(ass - tot_amt) < 1:
                    net = ass - disc
                    it["AssAmt"] = net
                    it["PreTaxVal"] = net
                    it["TotAmt"] = net

                    # Recalculate tax on net value
                    gst_rt = it.get("GstRt") or 0
                    if gst_rt > 0:
                        half_rate = gst_rt / 2
                        cgst = round(net * half_rate / 100, 2)
                        sgst = round(net * half_rate / 100, 2)
                        igst = it.get("IgstAmt") or 0
                        if igst == 0:
                            it["CgstAmt"] = cgst
                            it["SgstAmt"] = sgst
                        it["TotItemVal"] = round(
                            net + (it.get("CgstAmt") or 0)
                            + (it.get("SgstAmt") or 0)
                            + (it.get("IgstAmt") or 0), 2
                        )

            # Recalculate ValDtls totals from corrected items
            new_ass = sum(it.get("AssAmt") or 0 for it in items)
            new_cgst = sum(it.get("CgstAmt") or 0 for it in items)
            new_sgst = sum(it.get("SgstAmt") or 0 for it in items)
            new_disc = sum(it.get("Discount") or 0 for it in items)

            # Only update if the recalculated values are closer to declared
            if abs(new_ass - declared_ass) < abs(item_ass_sum - declared_ass):
                val["AssVal"] = new_ass
                val["CgstVal"] = new_cgst
                val["SgstVal"] = new_sgst
                val["Discount"] = new_disc
                val["TotInvVal"] = round(
                    new_ass + new_cgst + new_sgst
                    + (val.get("IgstVal") or 0)
                    + (val.get("CesVal") or 0)
                    + (val.get("OthChrg") or 0)
                    - new_disc
                    + (val.get("RndOffAmt") or 0), 2
                )

    # Fix PreTaxVal: default to AssAmt when zero
    for it in items:
        if not it.get("PreTaxVal") and it.get("AssAmt"):
            it["PreTaxVal"] = it["AssAmt"]

    # ----- Phase 2: Regex enrichment from raw PDF text -----
    if raw_pdf_text:
        seller = einvoice.get("SellerDtls") or {}
        buyer = einvoice.get("BuyerDtls") or {}

        # Extract phone numbers if missing
        if not seller.get("Ph"):
            phones = _PHONE_RE.findall(raw_pdf_text[:3000])
            # Filter: at least 7 digits, skip GSTINs and dates
            valid_phones = [
                p.strip() for p in phones
                if sum(c.isdigit() for c in p) >= 7
                and not _GSTIN_RE.match(p)
                and len(p) < 40
            ]
            if valid_phones:
                seller["Ph"] = valid_phones[0]

        # Extract emails if missing
        if not seller.get("Em"):
            emails = _EMAIL_RE.findall(raw_pdf_text[:3000])
            if emails:
                seller["Em"] = emails[0]

    # ----- Phase 3: Buyer GSTIN validation -----
    if buyer_gstin_list:
        buyer = einvoice.get("BuyerDtls") or {}
        extracted_gstin = buyer.get("Gstin", "")

        # If extracted GSTIN not in the known list, find the correct one
        if extracted_gstin and extracted_gstin not in buyer_gstin_list:
            # Check if it's actually the seller's GSTIN (common LLM mistake)
            seller_gstin = (einvoice.get("SellerDtls") or {}).get("Gstin", "")
            if extracted_gstin == seller_gstin:
                # LLM put seller GSTIN in buyer field — try to find correct one
                # Match by state code (first 2 digits)
                state_code = extracted_gstin[:2] if len(extracted_gstin) >= 2 else ""
                matches = [g for g in buyer_gstin_list if g.startswith(state_code)]
                if len(matches) == 1:
                    buyer["Gstin"] = matches[0]
                elif matches:
                    buyer["Gstin"] = matches[0]  # best guess: first match

            # Also try: maybe the extracted GSTIN has an OCR error
            # Check if any list item differs by only 1-2 chars
            if buyer.get("Gstin") == extracted_gstin:  # still unchanged
                for known in buyer_gstin_list:
                    if len(known) == 15 and len(extracted_gstin) == 15:
                        diffs = sum(
                            1 for a, b in zip(known, extracted_gstin) if a != b
                        )
                        if diffs <= 2:
                            buyer["Gstin"] = known
                            break

        # Derive Pos from corrected GSTIN
        if buyer.get("Gstin") and not buyer.get("Pos"):
            buyer["Pos"] = buyer["Gstin"][:2]

    # ----- ShipDtls: copy from BuyerDtls when empty -----
    ship = einvoice.get("ShipDtls") or {}
    if not ship.get("Gstin") and not ship.get("LglNm"):
        buyer = einvoice.get("BuyerDtls") or {}
        if buyer.get("Gstin"):
            ship["Gstin"] = buyer["Gstin"]
            ship["LglNm"] = buyer.get("LglNm", "")
            ship["TrdNm"] = buyer.get("TrdNm", "")
            if not ship.get("Addr1"):
                ship["Addr1"] = buyer.get("Addr1", "")
                ship["Addr2"] = buyer.get("Addr2", "")
                ship["Loc"] = buyer.get("Loc", "")
                ship["Pin"] = buyer.get("Pin")
                ship["Stcd"] = buyer.get("Stcd", "")
    einvoice["ShipDtls"] = ship

    return einvoice


# ============================================================================
# AGENT 8: OUTPUT GENERATOR
# ============================================================================


class OutputGeneratorAgent(BaseAgent):
    """Final Output Generation Agent"""

    def __init__(self):
        super().__init__("output_generator")

    def run(
        self,
        output_folder: Path,
        extraction: dict,
        transformer: dict,
        decision: str,
        rejection_template: str | None = None,
        rejection_reason: str | None = None,
        rejection_phase: int | None = None,
    ) -> dict:
        print(" [8/9] Output Generator: Starting...")

        invoice = extraction.get("invoice", {})

        # Map decision to status
        status_map = {
            "ACCEPT": "Pending Payment",
            "REJECT": "Rejected",
            "SET_ASIDE": "To Verify",
            "ERROR": "To Verify",
        }
        invoice_status = status_map.get(decision, "To Verify")

        # Generate outcome message
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if decision == "ACCEPT":
            outcome = f"Invoice accepted for payment on {timestamp}"
        elif decision == "REJECT":
            reason = (
                rejection_template or rejection_reason or "Validation failed"
            )
            outcome = f"Invoice rejected on {timestamp}: {reason}"
        else:
            outcome = f"Invoice requires review as of {timestamp}"

        # Build GST e-Invoice JSON
        einvoice_data = build_einvoice_json(extraction)

        # Post-extraction validation and enrichment
        einvoice_data = validate_and_enrich_einvoice(einvoice_data)

        # Attach processing metadata
        output_data = {
            **einvoice_data,
            "_processing": {
                "invoice_type": "Normal",
                "invoice_status": invoice_status,
                "invoice_source": "Email",
                "decision": decision,
                "rejection_template": rejection_template,
                "rejection_reason": rejection_reason,
                "rejection_phase": f"Phase {rejection_phase}"
                if rejection_phase
                else None,
                "outcome_message": outcome,
                "currency": invoice.get("currency", "INR"),
            },
        }

        # Save output
        output_file = output_folder / "Postprocessing_Data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(
            f"  => Saved: Postprocessing_Data.json (Status: {invoice_status})"
        )

        # Save decision artifact
        decision_output = self.create_output(
            {
                "final_decision": decision,
                "decision_class": decision,  # ACCEPT, REJECT, SET_ASIDE — used by investigation agent
                "invoice_status": invoice_status,
                "rejection_template": rejection_template,
                "rejection_reason": rejection_reason,
                "rejection_phase": f"Phase {rejection_phase}"
                if rejection_phase
                else None,
            }
        )
        self.save_artifact(output_folder, "08_decision.json", decision_output)

        return output_data


# ============================================================================
# AGENT 9: AUDIT LOGGER
# ============================================================================


class AuditLoggerAgent(BaseAgent):
    """Audit Logging Agent"""

    def __init__(self):
        super().__init__("audit_logger")

    def run(
        self,
        output_folder: Path,
        source_folder: Path,
        decision: str,
        processing_time: float,
    ) -> dict:
        print(" [9/9] Audit Logger: Creating audit trail...")

        output = self.create_output(
            {
                "case_id": source_folder.name,
                "source_folder": str(source_folder),
                "output_folder": str(output_folder),
                "processing_summary": {
                    "decision": decision,
                    "processing_time_seconds": round(processing_time, 2),
                    "total_llm_calls": _get_metrics()["llm_calls"],
                    "total_tokens": _get_metrics()["total_tokens"]["prompt"]
                    + _get_metrics()["total_tokens"]["completion"],
                    "total_cost_usd": round(
                        _get_metrics()["total_cost_usd"], 6
                    ),
                },
                "artifacts": [
                    "01_classification.json",
                    "02_extraction.json",
                    "03_phase1_validation.json",
                    "04_phase2_validation.json",
                    "05_phase3_validation.json",
                    "06_phase4_validation.json",
                    "07_transformation.json",
                    "08_decision.json",
                    "09_audit_log.json",
                    "Postprocessing_Data.json",
                ],
            }
        )

        self.save_artifact(output_folder, "09_audit_log.json", output)
        return output


# ============================================================================
# ORCHESTRATOR
# ============================================================================


def _handle_phase_rejection(
    output_folder: Path,
    source_folder: Path,
    extraction: dict,
    phase_result: dict,
    phase_num: int,
    start_time: float,
) -> dict:
    """Handle rejection at a specific validation phase."""
    transformer = TransformerAgent()
    transformer_result = transformer.run(
        output_folder, extraction, phase_result
    )

    output_gen = OutputGeneratorAgent()
    output_gen.run(
        output_folder,
        extraction,
        transformer_result,
        "REJECT",
        phase_result.get("rejection_template"),
        phase_result.get("rejection_reason"),
        rejection_phase=phase_num,
    )

    audit = AuditLoggerAgent()
    audit.run(output_folder, source_folder, "REJECT", time.time() - start_time)
    return {"decision": "REJECT", "phase": phase_num}


def _run_pipeline(
    source_folder: Path,
    output_folder: Path,
    start_time: float,
) -> dict:
    """Run the main processing pipeline (agents 1-9). Returns result dict."""
    # Agent 1: Classification
    classifier = ClassifierAgent()
    classification = classifier.run(source_folder, output_folder)

    # Agent 2: Extraction
    extractor = ExtractorAgent()
    extraction = extractor.run(source_folder, output_folder, classification)

    # Agent 3: Phase 1 Validation
    phase1 = Phase1ValidatorAgent()
    phase1_result = phase1.run(output_folder, extraction)

    if phase1_result["decision"] == "REJECT":
        return _handle_phase_rejection(
            output_folder,
            source_folder,
            extraction,
            phase1_result,
            1,
            start_time,
        )

    # Agent 4: Phase 2 Validation
    phase2 = Phase2ValidatorAgent()
    phase2_result = phase2.run(output_folder, extraction, phase1_result)

    if phase2_result["decision"] == "REJECT":
        return _handle_phase_rejection(
            output_folder,
            source_folder,
            extraction,
            phase2_result,
            2,
            start_time,
        )

    # Agent 5: Phase 3 Validation
    phase3 = Phase3ValidatorAgent()
    phase3_result = phase3.run(output_folder, extraction, phase2_result)

    if phase3_result["decision"] == "REJECT":
        return _handle_phase_rejection(
            output_folder,
            source_folder,
            extraction,
            phase3_result,
            3,
            start_time,
        )

    # Agent 6: Phase 4 Validation
    phase4 = Phase4ValidatorAgent()
    phase4_result = phase4.run(output_folder, extraction, phase3_result)

    # Agent 7: Transformer
    transformer = TransformerAgent()
    transformer_result = transformer.run(
        output_folder, extraction, phase4_result
    )

    # Agent 8: Output Generator
    output_gen = OutputGeneratorAgent()
    output_gen.run(
        output_folder,
        extraction,
        transformer_result,
        phase4_result["decision"],
        phase4_result.get("rejection_template"),
        phase4_result.get("rejection_reason"),
        rejection_phase=4 if phase4_result["decision"] == "REJECT" else None,
    )

    # Agent 9: Audit Logger
    processing_time = time.time() - start_time
    audit = AuditLoggerAgent()
    audit.run(
        output_folder, source_folder, phase4_result["decision"], processing_time
    )

    return {
        "decision": phase4_result["decision"],
        "processing_time": processing_time,
        "output_folder": str(output_folder),
    }


def process_invoice(source_folder: Path) -> dict:
    """Main pipeline orchestrator"""
    _ensure_gcp_initialized()
    start_time = time.time()

    case_id = source_folder.name
    output_folder = get_output_folder(case_id)

    print(f"\n{'=' * 60}")
    print(f"Processing: {case_id}")
    print(f"{'=' * 60}")

    # Reset metrics
    _metrics_store["METRICS"] = {
        "llm_calls": 0,
        "total_tokens": {"prompt": 0, "completion": 0},
        "total_cost_usd": 0.0,
        "agent_breakdown": [],
    }

    try:
        return _run_pipeline(source_folder, output_folder, start_time)

    except Exception as e:
        print(f"\n   Error: {e}")
        traceback.print_exc()

        # Save error
        error_file = output_folder / "error.txt"
        error_file.write_text(f"Error: {e}\n\n{traceback.format_exc()}")

        audit = AuditLoggerAgent()
        audit.run(
            output_folder, source_folder, "ERROR", time.time() - start_time
        )

        return {"decision": "ERROR", "error": str(e)}


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="General Invoice Processing Agent"
    )
    parser.add_argument(
        "--base-dir", "-b", type=str, help="Base directory with case folders"
    )
    parser.add_argument(
        "--case", "-c", type=str, help="Single case folder path"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--num-cases", "-n", type=int, help="Limit number of cases"
    )

    args = parser.parse_args()

    # Load configuration
    load_config(args.config)

    print("=" * 60)
    print("GENERAL INVOICE PROCESSING AGENT")
    print("=" * 60)
    print(f"Output Dir: {OUTPUT_BASE_DIR}")
    print()

    # Collect cases
    if args.case:
        case_folders = [Path(args.case)]
    elif args.base_dir:
        base_path = Path(args.base_dir)
        if not base_path.exists():
            print(f"Error: Directory not found: {base_path}")
            sys.exit(1)
        case_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])
        if args.num_cases:
            case_folders = case_folders[: args.num_cases]
    else:
        print("Error: Specify --base-dir or --case")
        sys.exit(1)

    print(f"Processing {len(case_folders)} case(s)\n")

    # Process
    stats = {"total": len(case_folders), "accept": 0, "reject": 0, "error": 0}

    for idx, folder in enumerate(case_folders, 1):
        print(f"[{idx}/{len(case_folders)}]", end="")
        result = process_invoice(folder)

        decision = result.get("decision", "ERROR")
        if decision == "ACCEPT":
            stats["accept"] += 1
        elif decision == "REJECT":
            stats["reject"] += 1
        else:
            stats["error"] += 1

        print(f"   Result: {decision}\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:    {stats['total']}")
    print(f"Accepted: {stats['accept']}")
    print(f"Rejected: {stats['reject']}")
    print(f"Errors:   {stats['error']}")


if __name__ == "__main__":
    main()
