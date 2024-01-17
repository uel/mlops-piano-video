from google.cloud import storage
import torch
import denoising_diffusion_pytorch
import functions_framework
import io

@functions_framework.http
def hello_http(request):
    """
    Generates a single image based on the trained model.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The image generated.
    """
    request_json = request.get_json()
    if request_json and 'predict' in request_json:
        
        BUCKET_NAME = "piano-video"

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.get_blob('models/diffusion_model.pt')
        buffer = io.BytesIO(blob.download_as_bytes())
        diffusion = torch.load(buffer)
        image = diffusion.sample(batch_size=1) # generates 1 image

        return f'success'

    else:
        return f'Hello World!'

# BUCKET_NAME = "piano-video"
# MODEL_FILE = "diffusion_model.pt"

# client = storage.Client()
# bucket = client.get_bucket(BUCKET_NAME)
# blob = bucket.get_blob('models/diffusion_model.pt')
# buffer = io.BytesIO(blob.download_as_bytes())
# diffusion = torch.load(buffer)
# image = diffusion.sample(batch_size=1) # generates 1 image

# print(f"success")
