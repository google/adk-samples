"""Data ingestion script for Hazmat Co-Pilot."""

import glob
import os
import re

import gcsfs
from app.db import get_db_connection_string
from google import genai
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# Initialize Gemini Client
client = genai.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), vertexai=True)


class SDSMetadata(BaseModel):
    hazard_type: list[str]
    hazard_pictograms: list[str]
    information_density: str


def extract_sds_metadata(section_2_text: str) -> SDSMetadata:
    """Extracts metadata from Section 2 using Gemini."""
    prompt = f"""
    Analyze the following Safety Data Sheet (SDS) Section 2 text and extract:
    1. Hazard Types (e.g., flammable, corrosive, toxic, oxidizer, irritant).
    2. Hazard Pictograms mentioned or implied (e.g., flame, exclamation_mark, corrosion, skull_and_crossbones).
    3. Information Density: Respond with 'high' if it is very technical and detailed, or 'low' if it is mostly summary or simple.
    
    Section 2 Text:
    {section_2_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": SDSMetadata,
            },
        )
        import json

        data = json.loads(response.text)
        return SDSMetadata(**data)
    except Exception as e:
        print(f"Error extracting metadata with LLM: {e}")
        return SDSMetadata(
            hazard_type=[], hazard_pictograms=[], information_density="low"
        )


def get_embedding(text: str) -> list[float]:
    """Generates embedding for text using gemini-embedding-001."""
    response = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return response.embeddings[0].values


def init_status_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_ingestion_status (
                filename TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()


def check_status(engine, filename):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT status FROM data_ingestion_status WHERE filename = :f"),
            {"f": filename},
        ).fetchone()
        return result[0] if result else None


def update_status(engine, filename, status):
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO data_ingestion_status (filename, status, updated_at)
            VALUES (:f, :s, CURRENT_TIMESTAMP)
            ON CONFLICT (filename) DO UPDATE SET status = :s, updated_at = CURRENT_TIMESTAMP
        """),
            {"f": filename, "s": status},
        )
        conn.commit()


def split_sds_sections(text: str) -> dict[int, str]:
    """Splits SDS text into sections based on regex."""
    pattern = re.compile(r"(?i)Section\s+(1[0-6]|[1-9])\b[:\s-]+")
    matches = list(pattern.finditer(text))

    valid_matches = []
    for match in matches:
        sec_num = int(match.group(1))
        start_idx = match.start()
        context_start = max(0, start_idx - 20)
        context = text[context_start:start_idx].lower()
        if "refer to" not in context and "see" not in context:
            valid_matches.append(match)

    sections = {}
    for i in range(len(valid_matches)):
        match = valid_matches[i]
        sec_num = int(match.group(1))
        start_idx = match.start()

        if i + 1 < len(valid_matches):
            end_idx = valid_matches[i + 1].start()
        else:
            end_idx = len(text)

        sec_text = text[start_idx:end_idx].strip()

        if sec_num not in sections:
            sections[sec_num] = sec_text
        else:
            if len(sec_text) > len(sections[sec_num]):
                sections[sec_num] = sec_text

    return sections


def ingest_files(batch_size: int = 10):
    """Ingests files from data/sds or GCS into PostgreSQL."""
    try:
        connection_string = get_db_connection_string().strip()
    except Exception as e:
        print(f"Error fetching connection string: {e}")
        return

    # Initialize engine for state tracking
    engine = create_engine(connection_string)
    init_status_table(engine)

    # Initialize vector store
    async_connection_string = connection_string.replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    vector_store = PGVectorStore(
        connection_string=connection_string,
        async_connection_string=async_connection_string,
        table_name="sds_vectors_3072",
        embed_dim=3072,
    )

    gcs_bucket = os.environ.get("GCS_BUCKET")

    if gcs_bucket:
        print(f"Using GCS bucket: {gcs_bucket}")
        fs = gcsfs.GCSFileSystem()
        files = fs.glob(f"gs://{gcs_bucket}/*.pdf")
    else:
        data_dir = "data/sds"
        print(f"Using local directory: {data_dir}")
        files = glob.glob(os.path.join(data_dir, "*.pdf"))

    if not files:
        print("No PDF files found.")
        return

    # Filter out already processed files
    target_file = os.environ.get("TARGET_FILE")
    files_to_process = []
    for f in files:
        filename = os.path.basename(f)
        if target_file and filename != target_file:
            continue
        status = check_status(engine, filename)
        # Skip SUCCESS and FAILED files to unblock pipeline
        if status not in ["SUCCESS", "FAILED"]:
            files_to_process.append(f)
        else:
            print(f"Skipping {filename} (already ingested)")

    print(f"Found {len(files_to_process)} files to process.")

    # Apply batch size
    files_to_process = files_to_process[:batch_size]
    print(f"Processing batch of {len(files_to_process)} files.")

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        chemical_id = filename.replace("_sds.pdf", "")
        print(f"Processing {chemical_id}...")
        update_status(engine, filename, "PROCESSING")

        local_path = file_path
        if gcs_bucket:
            local_path = f"/tmp/{filename}"
            print(f"Downloading {file_path} to {local_path}...")
            try:
                fs.get(file_path, local_path)
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
                update_status(engine, filename, "FAILED")
                continue

        try:
            print(f"Parsing PDF with pypdf: {local_path}")
            import pypdf
            from llama_index.core import Document

            pdf_reader = pypdf.PdfReader(local_path)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            text = text.replace("\x00", "")
            sections = split_sds_sections(text)
            print(f"Split into {len(sections)} sections.")

            # Extract metadata from Section 2
            sds_metadata = SDSMetadata(
                hazard_type=[], hazard_pictograms=[], information_density="low"
            )
            if 2 in sections:
                print("Extracting metadata from Section 2...")
                sds_metadata = extract_sds_metadata(sections[2])
                print(f"Extracted metadata: {sds_metadata}")

            documents = []
            for sec_num, sec_text in sections.items():
                persona_affinity = "general"
                if sec_num in [4, 5, 6, 7, 8, 9]:
                    persona_affinity = "workplace"
                elif sec_num in [2, 3, 15]:
                    persona_affinity = "regulatory"

                documents.append(
                    Document(
                        text=sec_text,
                        metadata={
                            "section_id": str(sec_num),
                            "chemical_id": chemical_id,
                            "persona_affinity": persona_affinity,
                            "hazard_type": sds_metadata.hazard_type,
                            "hazard_pictograms": sds_metadata.hazard_pictograms,
                            "information_density": sds_metadata.information_density,
                        },
                    )
                )

            if not documents:
                print("No sections found, fallback to full text.")
                documents = [
                    Document(
                        text=text,
                        metadata={
                            "chemical_id": chemical_id,
                            "persona_affinity": "general",
                            "hazard_type": [],
                            "hazard_pictograms": [],
                            "information_density": "low",
                        },
                    )
                ]

            parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents)

            print(f"Generated {len(nodes)} nodes. Calculating embeddings...")
            for node in nodes:
                try:
                    node.embedding = get_embedding(node.get_content())
                except Exception as e:
                    print(f"Error getting embedding for node: {e}")
                    continue

            try:
                vector_store.add(nodes)
                print(f"Successfully ingested {chemical_id}")
                update_status(engine, filename, "SUCCESS")
            except Exception as e:
                print(f"Error saving to vector store: {e}")
                update_status(engine, filename, "FAILED")
                print("Critical error saving to vector store. Aborting.")
                raise e

        except Exception as e:
            print(f"Error reading/parsing {chemical_id}: {e}")
            update_status(engine, filename, "FAILED")
            continue
        finally:
            if gcs_bucket and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as e:
                    print(f"Error removing temp file {local_path}: {e}")


if __name__ == "__main__":
    batch_size = int(os.environ.get("BATCH_SIZE", "10"))
    ingest_files(batch_size=batch_size)
