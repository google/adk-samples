# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.11-slim

RUN pip install --no-cache-dir uv==0.8.13

# Create non-root user for least-privilege container runtime security
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

WORKDIR /code

# Pre-sync dependencies for optimal Docker layer caching
COPY ./pyproject.toml ./README.md ./uv.lock* ./
RUN uv sync --frozen --no-install-project

# Copy application source and complete installation
COPY ./financial_advisor ./financial_advisor
RUN uv sync --frozen

# Grant write permissions for ADK browser runtime config to avoid PermissionError on startup
RUN chmod -R a+w /code/.venv/lib/python3.11/site-packages/google/adk/cli/browser/assets/config 2>/dev/null || true

# Run as non-root user
USER appuser

ENV PATH="/code/.venv/bin:$PATH"
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

ARG AGENT_VERSION=0.0.0
ENV AGENT_VERSION=${AGENT_VERSION}

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "financial_advisor.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "10"]

