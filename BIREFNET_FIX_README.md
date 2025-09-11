# BiRefNet 配置修复说明

## 问题描述

在使用BiRefNet模型时，可能会遇到以下错误：
```
'Config' object has no attribute 'is_encoder_decoder'
```

这是因为transformers库期望配置对象具有`is_encoder_decoder`属性，但BiRefNet的配置类中缺少这个属性。

## 解决方案

本插件已经内置了动态修复方案，无需手动修改models文件夹中的文件。

### 自动修复机制

插件会在加载BiRefNet模型时自动应用以下修复：

1. **动态检测配置类**：自动检测BiRefNet模型目录中的配置类
2. **运行时补丁**：在模型加载前动态添加缺失的`is_encoder_decoder`属性
3. **兼容性保证**：确保与transformers库的兼容性

### 修复的配置类

- `BiRefNetConfig` (在BiRefNet_config.py中)
- `Config` (在birefnet.py中)

### 支持的模型版本

- BiRefNet-General
- BiRefNet_HR
- BiRefNet_HR-matting
- BiRefNet_dynamic
- RMBG-2.0

## 使用方法

无需任何额外操作，插件会自动处理配置修复。当您使用BiRefNet节点时：

1. 插件会自动检测模型配置
2. 应用必要的修复补丁
3. 正常加载模型

## 技术细节

修复通过以下方式实现：

```python
def apply_birefnet_config_patch(self, model_path):
    """动态修复BiRefNet配置类缺少is_encoder_decoder属性的问题"""
    # 动态导入配置模块
    # 检查并修复配置类
    # 添加缺失的属性
```

这种方法的优势：
- ✅ 无需修改models文件夹
- ✅ 自动适配不同版本的BiRefNet模型
- ✅ 不影响原始模型文件
- ✅ 向后兼容

## 故障排除

如果仍然遇到问题，请检查：

1. 模型是否正确下载到models/BiRefNet/目录
2. 模型文件是否完整
3. 是否有足够的磁盘空间和内存

## 更新日志

- v1.0: 添加动态配置修复功能
- 自动处理BiRefNet模型的transformers兼容性问题
