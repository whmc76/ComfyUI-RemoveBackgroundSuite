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
sys.path.append(os.path.join(os.path.dirname(__file__), 'BiRefNet_v2'))

def get_files(path, extensions):
    files = {}
    for file in os.listdir(path):
        if file.endswith(extensions):
            files[file] = os.path.join(path, file)
    return files

def scan_model():
    model_path = os.path.join(folder_paths.models_dir, 'BiRefNet')
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
    mask = tensor2pil(mask)
    mask = np.array(mask)
    kernel = np.ones((erode, erode), np.uint8)
    erode_mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((dilate, dilate), np.uint8)
    dilate_mask = cv2.dilate(mask, kernel, iterations=1)
    trimap = np.zeros_like(mask)
    trimap[erode_mask > 0.5] = 1
    trimap[dilate_mask > 0.5] = 0.5
    return Image.fromarray(trimap)

def generate_VITMatte(image, trimap, local_files_only=False, device='cuda', max_megapixels=2.0):
    from transformers import AutoModelForImageSegmentation
    model_path = os.path.join(folder_paths.models_dir, 'BiRefNet', 'VITMatte')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="ZhengPeng7/VITMatte", local_dir=model_path, ignore_patterns=["*.md", "*.txt"])
    model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    image = np.array(image)
    trimap = np.array(trimap)
    image = cv2.resize(image, (1024, 1024))
    trimap = cv2.resize(trimap, (1024, 1024))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    trimap = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0).float()
    image = image.to(device)
    trimap = trimap.to(device)
    with torch.no_grad():
        pred = model(image, trimap)
    pred = pred.cpu().numpy().squeeze()
    pred = cv2.resize(pred, (image.shape[3], image.shape[2]))
    return Image.fromarray((pred * 255).astype(np.uint8))

def histogram_remap(mask, black_point, white_point):
    mask = tensor2pil(mask)
    mask = adjust_levels(mask, black_point, white_point)
    return torch.from_numpy(np.array(mask)).unsqueeze(0)

def log(message, message_type='info'):
    print(f"[{message_type.upper()}] {message}") 