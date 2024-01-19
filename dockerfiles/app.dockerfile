# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements_app.txt requirements_app.txt

RUN pip install --upgrade pip
RUN pip install -r requirements_app.txt --no-cache-dir

EXPOSE $PORT
CMD git clone https://github.com/uel/mlops-piano-video.git && cd mlops-piano-video && dvc pull models/tiny && uvicorn app.main:app --port $PORT --workers 1 main:app