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
            width, height = orig_image.size
            max_pixels = int(max_megapixels * 1_048_576)
            orig_pixels = width * height
            patch_size = 32
            if orig_pixels > max_pixels:
                scale = (max_pixels / orig_pixels) ** 0.5
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
            else:
                new_width, new_height = width, height
            new_width = (new_width // patch_size) * patch_size
            new_height = (new_height // patch_size) * patch_size
            new_width = max(patch_size, new_width)
            new_height = max(patch_size, new_height)
            inference_image_size = (new_width, new_height)
            log(f"[TransparentBackgroundUltra_RBS] 原始尺寸: {width}x{height}, 实际推理尺寸: {inference_image_size[0]}x{inference_image_size[1]}", message_type='info')
            resized_image = orig_image.resize(inference_image_size, Image.BILINEAR)
            ret_image = remover.process(resized_image, type='rgba')
            # 推理后还原为原图尺寸
            ret_image = ret_image.resize(orig_image.size, Image.BILINEAR)
            _mask = ret_image.split()[3]
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))
        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

# ======================== V3 新增：支持 BiRefNet-Dynamic ========================

class BiRefNetUltraV3_RBS:
    birefnet_model_repos = {
        "BiRefNet-General": "ZhengPeng7/BiRefNet",
        "RMBG-2.0": "briaai/RMBG-2.0",
        "BiRefNet_dynamic": "ZhengPeng7/BiRefNet_dynamic",
        "BiRefNet_HR": "ZhengPeng7/BiRefNet_HR",
        "BiRefNet_HR-matting": "ZhengPeng7/BiRefNet_HR-matting"
    }

    def __init__(self):
        self.NODE_NAME = 'BiRefNetUltraV3_RBS'
        self.model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        device_list = ['cuda', 'cpu']
        model_list = list(cls.birefnet_model_repos.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "version": (model_list, {"default": model_list[0]}),
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

    def load_birefnet_model(self, version):
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
                birefnet = BiRefNet(bb_pretrained=False)
                state_dict = torch.load(old_model_path, map_location='cpu', weights_only=True)
                state_dict = check_state_dict(state_dict)
                birefnet.load_state_dict(state_dict)
                return birefnet
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

        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        return model

    def birefnet_ultra_v3(self, image, version, device, max_megapixels):
        ret_images = []
        ret_masks = []
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        # 模型缓存，避免重复加载
        if version not in self.model_cache:
            self.model_cache[version] = self.load_birefnet_model(version)
        birefnet_model = self.model_cache[version]
        birefnet_model.to(device)
        birefnet_model.eval()
        comfy_pbar = ProgressBar(len(image))
        tqdm_pbar = tqdm.tqdm(total=len(image), desc="Processing BiRefNetV3")
        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            width, height = orig_image.size
            max_pixels = int(max_megapixels * 1_048_576)
            orig_pixels = width * height
            patch_size = 32  # 保证分辨率为32的倍数
            # 动态调整分辨率
            if orig_pixels > max_pixels:
                scale = (max_pixels / orig_pixels) ** 0.5
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
            else:
                new_width, new_height = width, height
            # 向下取整为patch_size的倍数
            new_width = (new_width // patch_size) * patch_size
            new_height = (new_height // patch_size) * patch_size
            new_width = max(patch_size, new_width)
            new_height = max(patch_size, new_height)
            inference_image_size = (new_width, new_height)
            log(f"[BiRefNetUltraV3_RBS] 原始尺寸: {width}x{height}, 实际推理尺寸: {inference_image_size[0]}x{inference_image_size[1]}", message_type='info')
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
            # 先还原到inference_image_size，再还原到原图尺寸
            _mask = pred_pil.resize(inference_image_size, Image.BILINEAR)
            _mask = _mask.resize(orig_image.size, Image.BILINEAR)
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

class ProcessDetails_RBS:
    def __init__(self):
        self.NODE_NAME = 'ProcessDetails_RBS'
        self.vitmatte_model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "detail_method": (["VITMatte", "VITMatte(local)", "PyMatting", "GuidedFilter"], {"default": "VITMatte"}),
                "detail_erode": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "detail_dilate": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "process_details"
    CATEGORY = 'RemoveBackgroundSuite'

    def process_details(self, image, mask, detail_method, detail_erode, detail_dilate, black_point, white_point, device, max_megapixels):
        ret_images = []
        ret_masks = []
        
        for i, m in zip(image, mask):
            orig_image = tensor2pil(i)
            orig_mask = tensor2pil(m)
            
            if detail_method.startswith("VITMatte"):
                local_files_only = detail_method == "VITMatte(local)"
                trimap = generate_VITMatte_trimap(orig_mask, detail_erode, detail_dilate)
                processed_mask = generate_VITMatte(orig_image, trimap, local_files_only, device, max_megapixels)
            else:
                # 计算目标尺寸
                width, height = orig_image.size
                max_pixels = int(max_megapixels * 1_048_576)
                orig_pixels = width * height
                patch_size = 32  # 保证分辨率为32的倍数
                
                if orig_pixels > max_pixels:
                    scale = (max_pixels / orig_pixels) ** 0.5
                    new_width = max(1, int(width * scale))
                    new_height = max(1, int(height * scale))
                else:
                    new_width, new_height = width, height
                    
                # 向下取整为patch_size的倍数
                new_width = (new_width // patch_size) * patch_size
                new_height = (new_height // patch_size) * patch_size
                new_width = max(patch_size, new_width)
                new_height = max(patch_size, new_height)
                inference_image_size = (new_width, new_height)
                
                log(f"[ProcessDetails_RBS] 原始尺寸: {width}x{height}, 实际推理尺寸: {inference_image_size[0]}x{inference_image_size[1]}", message_type='info')
                
                # 调整图像大小
                resized_image = orig_image.resize(inference_image_size, Image.BILINEAR)
                resized_mask = orig_mask.resize(inference_image_size, Image.BILINEAR)
                
                if detail_method == "PyMatting":
                    trimap = generate_VITMatte_trimap(resized_mask, detail_erode, detail_dilate)
                    try:
                        from pymatting import estimate_alpha_lkm
                    except ImportError:
                        raise RuntimeError("请先安装 pymatting 库: pip install pymatting scikit-image")
                    import numpy as np
                    image_np = np.array(resized_image.convert('RGB'))
                    trimap_np = np.array(trimap.convert('L')) / 255.0
                    # PyMatting只用LKM算法，参数风格与LayerStyle_Advance一致
                    alpha = estimate_alpha_lkm(image_np, trimap_np)
                    processed_mask = Image.fromarray((alpha * 255).astype(np.uint8))
                else:  # GuidedFilter
                    processed_mask = mask_edge_detail(pil2tensor(resized_image), image2mask(resized_mask), detail_erode, black_point, white_point)
                    processed_mask = tensor2pil(processed_mask)
                
                # 还原到原始尺寸
                processed_mask = processed_mask.resize(orig_image.size, Image.BILINEAR)
            
            # 应用处理后的蒙版到图像
            processed_image = RGB2RGBA(orig_image, processed_mask)
            
            ret_images.append(pil2tensor(processed_image))
            ret_masks.append(image2mask(processed_mask))
        
        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "TransparentBackgroundUltra_RBS": TransparentBackgroundUltra_RBS,
    "BiRefNetUltraV3_RBS": BiRefNetUltraV3_RBS,
    "MaskProcessDetails_RBS": ProcessDetails_RBS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparentBackgroundUltra_RBS": "Transparent Background Ultra (RBS)",
    "BiRefNetUltraV3_RBS": "BiRefNet Ultra V3 (RBS)",
    "MaskProcessDetails_RBS": "Mask Process Details (RBS)"
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