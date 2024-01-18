# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements_app.txt requirements_app.txt 
COPY /dockerfiles/app.sh /app.sh

RUN pip install --upgrade pip
RUN pip install -r requirements_app.txt --no-cache-dir

EXPOSE 8080
RUN chmod +x /app.sh

ENTRYPOINT ["/app.sh"]