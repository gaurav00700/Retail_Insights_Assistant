FROM python:3.11-slim

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     wget \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose backend port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
