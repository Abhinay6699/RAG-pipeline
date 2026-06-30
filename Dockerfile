FROM python:3.12-slim

# Set environment variables to avoid writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools for faiss-cpu and other potential C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install dependencies first (for docker cache layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose port (used for local testing)
EXPOSE 5000

# Start the application with Gunicorn, binding to the platform's dynamic PORT (or 5000 fallback)
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 app:app
