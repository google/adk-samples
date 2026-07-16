FROM python:3.12-slim

# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

WORKDIR /app
COPY . .

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip
# Install uv
RUN pip install --no-cache-dir uv>=0.7.19

# Copy only the dependency files to leverage Docker layer caching
COPY uv.lock pyproject.toml ./

# Install project dependencies, sync to uv's lockfile
RUN uv sync --frozen

# Expose the port
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
