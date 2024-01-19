from google.cloud import storage
import torch
import denoising_diffusion_pytorch
import functions_framework
import io
from PIL import Image
import base64

@functions_framework.http
def generate_image(request):
    """
    Generates a single image based on the trained model.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The image generated.
    """
    BUCKET_NAME = "piano-video"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob('models/tiny/diffusion_model.pt')
    buffer = io.BytesIO(blob.download_as_bytes())
    diffusion = torch.load(buffer, map_location=torch.device('cpu'))
    image = diffusion.sample(batch_size=1) # generates 1 image
    
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image.detach().numpy()
    image = (image * 255).astype('uint8')
    image = Image.fromarray(image)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    return "<img src='data:image/png;base64," + img_str.decode('utf-8') + "'/>"
    
      