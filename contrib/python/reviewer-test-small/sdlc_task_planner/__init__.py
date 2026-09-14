import os

import google.auth
from dotenv import load_dotenv

from . import agent

load_dotenv()

_, project_id = google.auth.default()
if project_id:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-east1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
