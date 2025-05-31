# ComfyUI-RemoveBackgroundSuite

这是一个 ComfyUI 插件，专注于实现各类高质量背景移除功能，支持多种 SOTA 算法和细节处理。

## 节点说明

### 1. LoadBiRefNetModel_RBS
- **功能**：加载本地 BiRefNet 模型权重。
- **参数**：
  - `model`：选择本地模型文件（.pth）。
- **输出**：`birefnet_model`（供后续节点使用）

### 2. LoadBiRefNetModelV2_RBS
- **功能**：自动下载并加载 BiRefNet 新版模型（支持 Huggingface 仓库）。
- **参数**：
  - `version`：选择模型版本（如 BiRefNet-General、RMBG-2.0）。
- **输出**：`birefnet_model`（供后续节点使用）

### 3. BiRefNetUltraV2_RBS
- **功能**：使用 BiRefNet Ultra V2 进行高质量背景移除。
- **参数**：
  - `image`：输入图片（支持批量）。
  - `birefnet_model`：已加载的模型。
  - `detail_method`：细节处理方式（VITMatte、PyMatting、GuidedFilter等）。
  - `detail_erode`/`detail_dilate`：腐蚀/膨胀参数，影响边缘细节。
  - `black_point`/`white_point`：黑白场，调整掩码对比度。
  - `process_detail`：是否进行细节处理。
  - `device`：推理设备（cuda/cpu）。
  - `max_megapixels`：最大处理分辨率。
- **输出**：
  - `image`：去背景后的 RGBA 图片
  - `mask`：前景掩码

### 4. TransparentBackgroundUltra_RBS
- **功能**：将图片背景转换为透明，支持多种细节处理。
- **参数**：
  - `image`：输入图片。
  - `model`：选择本地模型。
  - 其余参数同上。
- **输出**：
  - `image`：透明背景图片
  - `mask`：前景掩码

## 典型用法
1. 用 `LoadBiRefNetModel_RBS` 或 `LoadBiRefNetModelV2_RBS` 加载模型。
2. 用 `BiRefNetUltraV2_RBS` 进行背景移除。
3. 可选：用 `TransparentBackgroundUltra_RBS` 进一步处理透明背景。

## 注意事项
- 请将模型文件放在 `ComfyUI/models/BiRefNet/pth/` 目录下，或使用新版节点自动下载。
- 推荐使用 CUDA 设备以获得更快推理速度。
- 细节处理方法对边缘质量有显著影响，可根据实际需求调整。
- 插件所有节点均归类于 `RemoveBackgroundSuite`，便于统一管理。

## 依赖安装
```bash
pip install -r requirements.txt
```

## 常见问题
- **模型下载失败**：请检查网络连接或手动下载模型放入指定目录。
- **推理慢/显存不足**：可适当降低 `max_megapixels` 或切换到 CPU。
- **节点不显示**：请确认插件已放入 `custom_nodes` 目录并重启 ComfyUI。

---
如有更多问题请参考原项目文档或在 Issues 区反馈。