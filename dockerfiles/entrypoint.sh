#!/bin/bash

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "Error: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set."
    exit 1
fi

: "${BRANCH:=main}"

git config --global credential.helper store
echo "https://${GH_USER}:${GH_TOKEN}@github.com" > ~/.git-credentials
git config --global user.email "<>"
git config --global user.name "gcloud"

git clone https://github.com/uel/mlops-piano-video.git /mlops-piano-video -b "$BRANCH"

mkdir -p /mlops-piano-video/keys
echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /mlops-piano-video/keys/piano-video-99a62456b80f.json
export GOOGLE_APPLICATION_CREDENTIALS=/mlops-piano-video/keys/piano-video-99a62456b80f.json

cd /mlops-piano-video
pip install -r requirements.txt --no-cache-dir
dvc pull
mkdir -p models
python -u piano_video/train_model.py
dvc add models
dvc push
git add models.dvc
git commit -m "model update"
git push