# Base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY dockerfiles/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
