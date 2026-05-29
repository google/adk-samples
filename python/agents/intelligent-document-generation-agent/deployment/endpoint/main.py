# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import timedelta

import functions_framework
import google.auth
import markdown
from google.auth.transport import requests as google_requests
from google.cloud import storage
from jinja2 import BaseLoader, Environment
from weasyprint import HTML


def get_gcs_client():
    return storage.Client()


def convert_markdown_to_pdf(markdown_text: str) -> bytes:
    # Convert Markdown to HTML
    html_content = markdown.markdown(
        markdown_text, extensions=["extra", "codehilite", "tables"]
    )

    # Jinja2 template to provide basic styling
    template_source = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Document</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
            h1, h2, h3 { color: #333; }
            code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
            pre { background-color: #f4f4f4; padding: 10px; overflow-x: auto; border-radius: 4px; }
            pre code { background-color: transparent; padding: 0; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        {{ content }}
    </body>
    </html>
    """
    env = Environment(loader=BaseLoader())
    template = env.from_string(template_source)
    rendered_html = template.render(content=html_content)

    # Use weasyprint to convert HTML to PDF
    pdf_bytes = HTML(string=rendered_html).write_pdf()
    return pdf_bytes


def generate_signed_url(bucket_name: str, blob_name: str) -> str:
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    target_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials, project_id = google.auth.default(scopes=target_scopes)
    auth_request = google_requests.Request()
    credentials.refresh(auth_request)

    sa_email = credentials.service_account_email

    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        service_account_email=sa_email,
        access_token=credentials.token,
    )

    return url


def process_gcs_uri(gcs_uri: str) -> str:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI.")

    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download markdown text
    markdown_text = blob.download_as_text()

    # Convert to PDF
    pdf_bytes = convert_markdown_to_pdf(markdown_text)

    # Upload PDF
    pdf_blob_name = blob_name.rsplit(".", 1)[0] + ".pdf"
    if pdf_blob_name == blob_name:
        pdf_blob_name += ".pdf"

    pdf_blob = bucket.blob(pdf_blob_name)
    pdf_blob.upload_from_string(pdf_bytes, content_type="application/pdf")

    # Generate Signed URL
    signed_url = generate_signed_url(bucket_name, pdf_blob_name)
    return signed_url


@functions_framework.http
def get_markdown_url(request):
    gcs_uri = request.args.get("gcs_uri")
    if not gcs_uri:
        return "Missing required parameters.", 400

    try:
        signed_url = process_gcs_uri(gcs_uri)
        return signed_url, 200

    except ValueError as parse_err:
        return str(parse_err), 400
    except Exception as e:
        import traceback

        traceback.print_exc()
        return str(e), 500
