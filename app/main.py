import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torchvision, torch

app = FastAPI()

@app.get('/')
def root():
    return "Hello! Go to /predict to predict"

@app.get("/predict", response_class=StreamingResponse)
def show_image():
    tensor = torch.rand((3,1000,1000)) # read the tensor from disk or whatever
    image = torchvision.transforms.ToPILImage()(tensor)
    return_image = io.BytesIO()
    image.save(return_image, "JPEG")
    return_image.seek(0)
    return StreamingResponse(content=return_image, media_type="image/jpeg")