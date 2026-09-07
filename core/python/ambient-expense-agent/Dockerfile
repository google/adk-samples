FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY expense_agent/ ./expense_agent/

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "python", "expense_agent/fast_api_app.py"]
