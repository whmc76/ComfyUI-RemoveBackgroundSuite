import torch
import numpy as np
from PIL import Image
import folder_paths
import os
from .model_utils import load_model, preprocess_image, postprocess_mask, apply_transparency
from .imagefunc import *
from comfy.utils import ProgressBar
import tqdm
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import sys
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), 'BiRefNet'))
from .BiRefNet.models.birefnet import BiRefNet
from .BiRefNet.utils import check_state_dict

# 获取本地所有BiRefNet模型文件
# 返回字典：{模型文件名: 路径}
def get_models():
    model_path = os.path.join(folder_paths.models_dir, 'BiRefNet', 'pth')
    model_ext = [".pth"]
    model_dict = get_files(model_path, model_ext)
    return model_dict

# 透明背景超强节点
class TransparentBackgroundUltra_RBS:
    def __init__(self):
        self.NODE_NAME = 'TransparentBackgroundUltra_RBS'

    @classmethod
    def INPUT_TYPES(cls):
        device_list = ['cuda','cpu']
        def scan_transparent_models():
            import glob
            model_file_list = glob.glob(os.path.join(folder_paths.models_dir, "transparent-background") + '/*.pth')
            model_dict = {}
            for i in range(len(model_file_list)):
                _, __filename = os.path.split(model_file_list[i])
                model_dict[__filename] = model_file_list[i]
            return model_dict
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(scan_transparent_models().keys()),),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "transparent_background_ultra"
    CATEGORY = 'RemoveBackgroundSuite'
  
    def transparent_background_ultra(self, image, model, device, max_megapixels):
        import glob
        from transparent_background import Remover
        mode_dict = {"ckpt_base.pth": "base", "ckpt_base_nightly.pth": "base-nightly", "ckpt_fast.pth": "fast"}
        ret_images = []
        ret_masks = []
        model_file_list = glob.glob(os.path.join(folder_paths.models_dir, "transparent-background") + '/*.pth')
        model_dict = {}
        for i in range(len(model_file_list)):
            _, __filename = os.path.split(model_file_list[i])
            model_dict[__filename] = model_file_list[i]
        try:
            mode = mode_dict[model]
        except:
            mode = "base"
        remover = Remover(mode=mode, jit=False, device=device, ckpt=model_dict[model])
        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            ret_image = remover.process(orig_image, type='rgba')
            _mask = ret_image.split()[3]
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))
        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

# ======================== V3 新增：支持 BiRefNet-Dynamic ========================

class LoadBiRefNetModelV3_RBS:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        # 新增 dynamic 选项
        model_list = list(s.birefnet_model_repos.keys())
        return {
            "required": {
                "version": (model_list, {"default": model_list[0]}), # 选择模型版本
            },
        }

    RETURN_TYPES = ("BIREFNET_MODEL",)
    RETURN_NAMES = ("birefnet_model",)
    FUNCTION = "load_birefnet_model_v3"
    CATEGORY = 'RemoveBackgroundSuite'

    # 支持 dynamic、HR、HR-matting 模型
    birefnet_model_repos = {
        "BiRefNet-General": "ZhengPeng7/BiRefNet",
        "RMBG-2.0": "briaai/RMBG-2.0",
        "BiRefNet_dynamic": "ZhengPeng7/BiRefNet_dynamic",
        "BiRefNet_HR": "ZhengPeng7/BiRefNet_HR",
        "BiRefNet_HR-matting": "ZhengPeng7/BiRefNet_HR-matting"
    }

    def load_birefnet_model_v3(self, version):
        birefnet_path = os.path.join(folder_paths.models_dir, 'BiRefNet')
        os.makedirs(birefnet_path, exist_ok=True)
        model_path = os.path.join(birefnet_path, version)

        if version == "BiRefNet-General":
            old_birefnet_path = os.path.join(birefnet_path, 'pth')
            old_model = "BiRefNet-general-epoch_244.pth"
            old_model_path = os.path.join(old_birefnet_path, old_model)
            if os.path.exists(old_model_path):
                from .BiRefNet.models.birefnet import BiRefNet
                from .BiRefNet.utils import check_state_dict
                self.birefnet = BiRefNet(bb_pretrained=False)
                self.state_dict = torch.load(old_model_path, map_location='cpu', weights_only=True)
                self.state_dict = check_state_dict(self.state_dict)
                self.birefnet.load_state_dict(self.state_dict)
                return (self.birefnet,)
        # 动态模型及HR、HR-matting模型下载
        if version in ["BiRefNet_dynamic", "BiRefNet_HR", "BiRefNet_HR-matting"] and not os.path.exists(model_path):
            log(f"Downloading {version} model...")
            from huggingface_hub import snapshot_download
            repo_id = self.birefnet_model_repos[version]
            snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt"])
        elif version == "RMBG-2.0" and not os.path.exists(model_path):
            log(f"Downloading RMBG-2.0 model...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="briaai/RMBG-2.0", local_dir=model_path, ignore_patterns=["*.md", "*.txt"])

        self.model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        return (self.model,)

class BiRefNetUltraV3_RBS:
    def __init__(self):
        self.NODE_NAME = 'BiRefNetUltraV3_RBS'

    @classmethod
    def INPUT_TYPES(cls):
        device_list = ['cuda', 'cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "birefnet_model": ("BIREFNET_MODEL",),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "birefnet_ultra_v3"
    CATEGORY = 'RemoveBackgroundSuite'

    # 主推理流程，兼容 dynamic 模型
    def birefnet_ultra_v3(self, image, birefnet_model, device, max_megapixels):
        ret_images = []
        ret_masks = []
        inference_image_size = (1024, 1024)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        birefnet_model.to(device)
        birefnet_model.eval()
        comfy_pbar = ProgressBar(len(image))
        tqdm_pbar = tqdm.tqdm(total=len(image), desc="Processing BiRefNetV3")
        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            transform_image = transforms.Compose([
                transforms.Resize(inference_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            inference_image = transform_image(orig_image).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = birefnet_model(inference_image)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            _mask = pred_pil.resize(inference_image_size)
            resize_sampler = Image.BILINEAR
            _mask = _mask.resize(orig_image.size, resize_sampler)
            brightness_image = ImageEnhance.Brightness(_mask)
            _mask = brightness_image.enhance(factor=1.08)
            _mask = image2mask(_mask)
            if not isinstance(_mask, Image.Image):
                _mask = tensor2pil(_mask)
            if _mask.mode != 'L':
                _mask = _mask.convert('L')
            if _mask.size != orig_image.size:
                _mask = _mask.resize(orig_image.size, Image.BILINEAR)
            ret_image = RGB2RGBA(orig_image, _mask)
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))
            comfy_pbar.update(1)
            tqdm_pbar.update(1)
        log(f"{self.NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "TransparentBackgroundUltra_RBS": TransparentBackgroundUltra_RBS,
    "LoadBiRefNetModelV3_RBS": LoadBiRefNetModelV3_RBS,
    "BiRefNetUltraV3_RBS": BiRefNetUltraV3_RBS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparentBackgroundUltra_RBS": "Transparent Background Ultra (RBS)",
    "LoadBiRefNetModelV3_RBS": "Load BiRefNet Model V3 (RBS)",
    "BiRefNetUltraV3_RBS": "BiRefNet Ultra V3 (RBS)"
}

def log_mask_info(tag, mask):
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
        log(f"[{tag}] mask(tensor) shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}, unique={np.unique(arr).size}")
    elif isinstance(mask, Image.Image):
        arr = np.array(mask)
        log(f"[{tag}] mask(PIL) shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}, unique={np.unique(arr).size}")
    else:
        log(f"[{tag}] mask类型未知: {type(mask)}") 