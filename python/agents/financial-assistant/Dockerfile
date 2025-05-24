# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION=1.7.1

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Copy pyproject.toml and poetry.lock* (if exists)
COPY pyproject.toml poetry.lock* /app/

# Generate poetry.lock and requirements.txt, then install with pip
RUN poetry lock --no-interaction --no-update \
    && poetry export -f requirements.txt --output requirements.txt --without-hashes --no-interaction --no-ansi \
    && pip install -r requirements.txt

# Copy the rest of the application code
# Important: This should come *after* dependency installation to leverage Docker caching
COPY financial_assistant/ /app/financial_assistant/
COPY .env.example /app/.env.example
# If there are other essential files or directories at the root of 'financial-assistant' project,
# ensure they are copied as well. e.g. if 'adk' CLI is part of the project structure and needed at runtime.
# COPY some_other_directory_or_file /app/some_other_directory_or_file/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the FastAPI application using Uvicorn
# The application is financial_advisor.main:app
# financial_advisor is the package/directory containing main.py
# app is the FastAPI instance in main.py
CMD ["uvicorn", "financial_advisor.main:app", "--host", "0.0.0.0", "--port", "8000"]
