# ComfyUI-RemoveBackgroundSuite
![image](https://github.com/user-attachments/assets/75fb08b6-184d-46e1-a4e6-3294dd94d66d)

> 基于 ComfyUI 的抠图套件，支持多种抠图模型和细节处理方式。

## 版本信息

- 当前版本：v1.2
- 更新日期：2024-06-XX

## 更新日志

### v1.2 (2024-06-XX)
- 节点界面极简化，移除所有细节处理相关参数和功能。
- TransparentBackgroundUltra_RBS 去除 WIP 标志。

### v1.1 (2024-03-21)
- 优化 BiRefNet 模型加载逻辑，支持 dynamic、HR、HR-matting 模型
- 优化 VITMatte 模型加载和推理流程
- 改进文档结构和说明

### v1.0 (2024-03-20)
- 初始版本发布
- 支持 BiRefNet 和 TransparentBackground 两种抠图模型

## 节点说明

### 1. LoadBiRefNetModelV3_RBS（推荐）
- **功能**：自动下载并加载 BiRefNet 最新模型，支持 BiRefNet_dynamic。
- **参数**：
  - `version`：选择模型版本（BiRefNet-General、RMBG-2.0、BiRefNet_dynamic）。
- **输出**：`birefnet_model`（供后续节点使用）

### 2. BiRefNetUltraV3_RBS（推荐）
- **功能**：使用 BiRefNet Ultra V3 进行高质量背景移除，支持 dynamic 动态模型。
- **参数**：
  - `image`：输入图片（支持批量）。
  - `birefnet_model`：已加载的模型（支持 dynamic）。
  - `device`、`max_megapixels`：详见节点界面。
- **输出**：
  - `image`：去背景后的 RGBA 图片
  - `mask`：前景掩码

### 3. TransparentBackgroundUltra_RBS
- **功能**：将图片背景转换为透明。
- **参数**：
  - `image`：输入图片。
  - `model`：选择本地模型。
  - `device`、`max_megapixels`：详见节点界面。
- **输出**：
  - `image`：透明背景图片
  - `mask`：前景掩码

## 典型用法
1. 用 `LoadBiRefNetModelV3_RBS` 加载模型（推荐选择 BiRefNet_dynamic）。
2. 用 `BiRefNetUltraV3_RBS` 进行背景移除。
3. 可选：用 `TransparentBackgroundUltra_RBS` 进一步处理透明背景。

## 注意事项
- 请将模型文件放在 `ComfyUI/models/transparent-background/` 目录下，或使用新版节点自动下载。
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

## 致谢
本插件大量借鉴和参考了 [ComfyUI_LayerStyle_Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance) 项目的设计与实现，特别感谢原作者 chflame163 的开源贡献！

如有更多问题请参考原项目文档或在 Issues 区反馈。
