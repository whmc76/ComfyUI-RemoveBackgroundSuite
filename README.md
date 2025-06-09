# ComfyUI-RemoveBackgroundSuite

A matting toolkit based on ComfyUI, supporting multiple matting models and detail processing methods.

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

### BiRefNetUltraV3_RBS
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
