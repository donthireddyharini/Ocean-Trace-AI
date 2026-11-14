FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies with PyTorch CPU wheel (for lightweight Docker containers)
RUN pip install --no-cache-dir -r requirements.txt

# Copy microservice code
COPY . .

# Hugging Face Spaces & Render standard port
EXPOSE 7860

ENV PORT=7860
ENV IMAGE_SIZE=512

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
