# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY /app /app
COPY /models /models
COPY requirements_app.txt requirements_app.txt 

RUN pip install --upgrade pip
RUN pip install -r requirements_app.txt

EXPOSE $PORT

CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1