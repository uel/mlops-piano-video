#!/bin/bash

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "Error: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set."
    exit 1
fi

git clone https://github.com/uel/mlops-piano-video.git /mlops-piano-video

mkdir -p /mlops-piano-video/keys
echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /mlops-piano-video/keys/piano-video-99a62456b80f.json
export GOOGLE_APPLICATION_CREDENTIALS=/mlops-piano-video/keys/piano-video-99a62456b80f.json

cd /mlops-piano-video
pip install -r requirements.txt --no-cache-dir
dvc remote add -d remote_storage gs://piano-video
dvc pull
python -u piano_video/train_model.py
