# ComfyUI-RemoveBackgroundSuite

**版本**: v1.2.4  
**ComfyUI 插件**: 抠图工具包

A matting toolkit based on ComfyUI, supporting multiple matting models and detail processing methods.

## 🆕 最新更新 (v1.2.4)

- **更新ComfyUI Registry配置**: 优化了pyproject.toml配置文件，符合最新的ComfyUI Registry发布规范
- **完善依赖管理**: 同步更新了requirements.txt文件，确保所有依赖项完整
- **增强元数据**: 添加了更详细的标签和描述信息，提升在Registry中的可发现性
- **图标支持**: 配置了项目图标URL，提升用户体验

### v1.2.3
- **自动修复BiRefNet兼容性问题**: 新增动态配置修复功能，自动解决BiRefNet模型与transformers库的兼容性问题
- **无需手动修改**: 用户安装插件后无需修改models文件夹中的任何文件
- **智能检测**: 自动检测并修复所有BiRefNet模型版本的配置问题
- **向后兼容**: 完全兼容现有工作流程，不影响原始模型文件

## Features

- **Multiple Models**: Support for various matting models including BiRefNet, RMBG, and more
- **Detail Processing**: Advanced mask processing capabilities for fine-tuning results
- **User-Friendly**: Simple and intuitive interface within ComfyUI
- **High Performance**: Optimized for both quality and speed

## Installation

1. Navigate to your ComfyUI's `custom_nodes` directory
2. Clone this repository:
```bash
git clone https://github.com/whmc76/ComfyUI-RemoveBackgroundSuite.git
```
3. Install dependencies:
```bash
cd ComfyUI-RemoveBackgroundSuite
pip install -r requirements.txt
```

## Usage

1. Start ComfyUI
2. The new nodes will appear in the node menu under the "RBS" category
3. Connect the nodes as needed in your workflow

## Models

The following models are supported:

- BiRefNet-General
- BiRefNet_dynamic
- BiRefNet_HR
- BiRefNet_HR-matting
- RMBG-2.0

## Nodes

### BiRefNetUltra_RBS
- **Input**: Image
- **Output**: Mask
- **Parameters**:
  - Model Version: Select from available BiRefNet models
  - Max Megapixels: Maximum image size for processing

### Transparent Background Ultra (RBS)
- **Input**: Image
- **Output**: Transparent Image
- **Parameters**:
  - Model Version: Select from available models
  - Max Megapixels: Maximum image size for processing

### Mask Process Details (RBS)
- **Input**: Mask
- **Output**: Processed Mask
- **Parameters**:
  - Detail Method: Choose from VITMatte, PyMatting, or GuidedFilter
  - Erode/Dilate: Control the trimap generation
  - Black/White Point: Adjust mask levels
  - Max Megapixels: Control processing resolution

## Changelog

### v1.2.1
- 完善了ComfyUI插件配置和元数据
- 添加了详细的插件描述和功能标签
- 优化了README.md的版本信息显示
- 改进了ComfyUI Registry的兼容性

### v1.2.0
- 为 BiRefNet Ultra (RBS) 和 Transparent Background Ultra (RBS) 节点添加了 mask 输入功能
- 新增 "自动适用原图遮罩" 参数，支持智能检测和使用输入的 mask
- 优化了节点名称，移除了 BiRefNetUltraV3_RBS 中的 "V3" 后缀
- 修复了输入 mask 应用时的翻转问题，确保输出结果正确
- 改进了批量处理逻辑，支持图像和 mask 的自动匹配

### v1.1.2
- 修复了模型加载路径的问题
- 改进了模型自动下载功能
- 添加了缺失的依赖项
- 优化了错误处理和日志记录

### v1.1.1
- Fixed VITMatte processing quality issues
- Optimized image size handling for different processing methods
- Improved mask processing workflow

### v1.1.0
- Optimized dependency management
- Removed version constraints for better compatibility
- Removed unused dependencies
- Improved code organization

### v1.0.0
- Initial release with core functionality
- Support for multiple matting models
- Basic mask processing capabilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the amazing framework
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) for the matting models
- [RMBG](https://github.com/briaai/RMBG-2.0) for the background removal model
