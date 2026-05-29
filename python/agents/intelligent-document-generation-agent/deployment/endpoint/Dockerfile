FROM debian:bullseye-slim

# Install system dependencies including Python 3.9 and WeasyPrint dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-cffi \
    python3-brotli \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libjpeg-dev \
    libopenjp2-7-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .

# Command to run the application using functions-framework
CMD exec functions-framework --target=get_markdown_url --port=${PORT:=8080}
