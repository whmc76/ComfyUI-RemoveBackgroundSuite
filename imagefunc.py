import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import folder_paths
import cv2
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import sys
import math
sys.path.append(os.path.join(os.path.dirname(__file__), 'BiRefNet'))

def get_files(path, extensions):
    if isinstance(extensions, list):
        extensions = tuple(extensions)
    files = {}
    for file in os.listdir(path):
        if file.endswith(extensions):
            files[file] = os.path.join(path, file)
    return files

def scan_model():
    model_path = os.path.join(folder_paths.models_dir, 'transparent-background')
    model_ext = [".pth"]
    model_dict = get_files(model_path, model_ext)
    return model_dict

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def RGB2RGBA(image, mask):
    image = image.convert('RGBA')
    image.putalpha(mask)
    return image

def adjust_levels(image, black_point, white_point):
    image = np.array(image)
    image = np.clip((image - black_point) / (white_point - black_point), 0, 1)
    return Image.fromarray((image * 255).astype(np.uint8))

def guided_filter_alpha(image, mask, radius):
    image = tensor2pil(image)
    mask = tensor2pil(mask)
    image = np.array(image)
    mask = np.array(mask)
    mask = cv2.ximgproc.guidedFilter(image, mask, radius, 1e-6)
    return torch.from_numpy(mask).unsqueeze(0)

def mask_edge_detail(image, mask, radius, black_point, white_point):
    image = tensor2pil(image)
    mask = tensor2pil(mask)
    image = np.array(image)
    mask = np.array(mask)
    mask = cv2.ximgproc.guidedFilter(image, mask, radius, 1e-6)
    mask = adjust_levels(Image.fromarray(mask), black_point, white_point)
    return torch.from_numpy(np.array(mask)).unsqueeze(0)

def generate_VITMatte_trimap(mask, erode, dilate):
    # 保证mask为0~255的uint8
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        mask = arr.squeeze()
    elif isinstance(mask, Image.Image):
        mask = np.array(mask)
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
    else:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    kernel = np.ones((erode, erode), np.uint8)
    erode_mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((dilate, dilate), np.uint8)
    dilate_mask = cv2.dilate(mask, kernel, iterations=1)
    trimap = np.zeros_like(mask, dtype=np.uint8)
    trimap[erode_mask > 250] = 255
    trimap[(dilate_mask > 0) & (erode_mask <= 250)] = 128
    return Image.fromarray(trimap)

def check_and_download_model(model_path, repo_id):
    model_path = os.path.join(folder_paths.models_dir, model_path)
    if not os.path.exists(model_path):
        print(f"Downloading {repo_id} model...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt", "onnx", ".git"])
    return model_path

class VITMatteModel:
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor

def load_VITMatte_model(model_name:str, local_files_only:bool=False) -> object:
    model_name = "vitmatte"
    model_repo = "hustvl/vitmatte-small-composition-1k"
    model_path  = check_and_download_model(model_name, model_repo)
    from transformers import VitMatteImageProcessor, VitMatteForImageMatting
    model = VitMatteForImageMatting.from_pretrained(model_path, local_files_only=local_files_only)
    processor = VitMatteImageProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    vitmatte = VITMatteModel(model, processor)
    return vitmatte

def generate_VITMatte(image, trimap, local_files_only=False, device='cuda', max_megapixels=2.0):
    import torch
    from PIL import Image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if trimap.mode != 'L':
        trimap = trimap.convert('L')
    max_megapixels *= 1048576
    width, height = image.size
    ratio = width / height
    target_width = math.sqrt(ratio * max_megapixels)
    target_height = target_width / ratio
    target_width = int(target_width)
    target_height = int(target_height)
    if width * height > max_megapixels:
        image = image.resize((target_width, target_height), Image.BILINEAR)
        trimap = trimap.resize((target_width, target_height), Image.BILINEAR)
    model_name = "hustvl/vitmatte-small-composition-1k"
    if device=="cpu":
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("vitmatte device is set to cuda, but not available, using cpu instead.")
            device = torch.device('cpu')
    vit_matte_model = load_VITMatte_model(model_name=model_name, local_files_only=local_files_only)
    vit_matte_model.model.to(device)
    inputs = vit_matte_model.processor(images=image, trimaps=trimap, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        predictions = vit_matte_model.model(**inputs).alphas
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    mask = tensor2pil(predictions).convert('L')
    mask = mask.crop((0, 0, image.width, image.height))
    if width * height > max_megapixels:
        mask = mask.resize((width, height), Image.BILINEAR)
    return mask

def histogram_remap(mask, black_point, white_point):
    mask = tensor2pil(mask)
    mask = adjust_levels(mask, black_point, white_point)
    return torch.from_numpy(np.array(mask)).unsqueeze(0)

def log(message, message_type='info'):
    print(f"[{message_type.upper()}] {message}") 