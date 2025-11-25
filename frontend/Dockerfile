FROM python:3.12-slim

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     wget \
    # curl \
    # && rm -rf /var/lib/apt/lists/*

COPY ./frontend /workdir/frontend

WORKDIR /workdir

# COPY requirements.txt .

RUN pip install -r ./frontend/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
