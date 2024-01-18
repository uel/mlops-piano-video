import io, os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torchvision, torch
import torch, denoising_diffusion_pytorch

app = FastAPI()
file_name = "diffusion_model.pt"
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tiny', file_name)

@app.get('/')
def root():
    return "Hello! Go to /predict to predict"

@app.get("/predict", response_class=StreamingResponse)
def predict():
    diffusion = torch.load(model_path)
    image = diffusion.sample(batch_size=1)
    image = image.squeeze(0)
    image = image.detach()
    image = torchvision.transforms.ToPILImage()(image)
    return_image = io.BytesIO()
    image.save(return_image, "JPEG")
    return_image.seek(0)
    return StreamingResponse(content=return_image, media_type="image/jpeg")