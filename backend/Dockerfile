FROM python:3.12-slim

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     wget \
# curl 
# && rm -rf /var/lib/apt/lists/*

COPY ./backend /workdir/backend

WORKDIR /workdir

# COPY requirements.txt .

RUN pip install -r ./backend/requirements.txt

# Expose backend port
EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
