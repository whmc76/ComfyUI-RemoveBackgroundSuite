import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class BiRefNet(nn.Module):
    def __init__(self):
        super(BiRefNet, self).__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 跳跃连接
        self.skip1 = nn.Conv2d(64, 64, 1)
        self.skip2 = nn.Conv2d(128, 128, 1)
        
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # 解码
        d3 = self.dec3(e3)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 = d3 + self.skip2(e2)
        
        d2 = self.dec2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = d2 + self.skip1(e1)
        
        d1 = self.dec1(d2)
        return d1

def load_model(model_path):
    """加载模型"""
    model = BiRefNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """预处理图像"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 转换为RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 调整大小为32的倍数
    width, height = image.size
    new_width = ((width + 31) // 32) * 32
    new_height = ((height + 31) // 32) * 32
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 转换为tensor
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def postprocess_mask(mask, original_size):
    """后处理掩码"""
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.float32)
    
    # 调整回原始大小
    if mask.shape != original_size:
        mask = Image.fromarray(mask)
        mask = mask.resize(original_size, Image.LANCZOS)
        mask = np.array(mask)
    
    return mask

def apply_transparency(image, mask, alpha=1.0):
    """应用透明效果"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 确保图像是RGBA模式
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 创建alpha通道
    alpha_channel = (mask * 255 * alpha).astype(np.uint8)
    alpha_channel = Image.fromarray(alpha_channel)
    
    # 应用alpha通道
    r, g, b, _ = image.split()
    image = Image.merge('RGBA', (r, g, b, alpha_channel))
    
    return image 