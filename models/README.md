# 本地模型

将兼容的 Depth Anything V2 Large 权重放在 `depth_anything_v2_large.onnx`。权重不入 Git。当前实现要求 RGB 输入及模型外 ImageNet 归一化，替换 ONNX 前请核对 [算法说明](../docs/usage.md)。V5 可通过 `--model` 指定其他位置。
