import io
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep all faces to check how many were detected
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE, keep_all=True, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def extract_embedding_from_image(image: Image.Image) -> np.ndarray:
    """
    Extract 512-d FaceNet embedding from a PIL Image.
    """
    face_tensors = mtcnn(image)

    if face_tensors is None or len(face_tensors) == 0:
        raise ValueError("No face detected in the image. Please submit another selfie.")

    if len(face_tensors) > 1:
        raise ValueError("Multiple faces detected in the image. Please ensure only one person is visible.")

    face_tensor = face_tensors[0].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = resnet(face_tensor)

    return embedding.cpu().numpy()[0]

def extract_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Extract embedding from raw image bytes.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise ValueError("Invalid image bytes received.")

    return extract_embedding_from_image(image)
