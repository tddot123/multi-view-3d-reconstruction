# Multi-View 3D Reconstruction and Visualization

**基于深度估计的多视角三维重建与可视化系统 · 正式 v2**

使用 C++17、OpenCV、ONNX Runtime 和 OpenGL/freeglut，从前、后、左、右、顶视图估计相对深度，重建平滑三角网格，并提供交互查看、PLY 和多角度检查图导出。当前算法采用轴对齐、近似正交投影，尚不包含相机标定或米制重建。

[仓库](https://github.com/tddot123/multi-view-3d-reconstruction) · [构建指南](docs/building.md) · [算法与运行参数](docs/usage.md) · [目录与迁移说明](docs/project-structure.md) · [版本记录](docs/releases.md) · [验证记录](docs/validation.md)

## 版本

- `codex/v1`：整理前的完整源码、文档、工程及样例快照，包含整理开始时尚未提交的 V5 工作，提交 `a32e2056d7f8a0f4b4aa3c27a774e0a72d8821c5`。
- `codex/v2`：在上述快照基础上整理目录、修正路径和构建配置的正式版本。
- 正式 **v1/v2** 与算法 **V4/V5** 是两套编号。正式 v1 已包含算法 V5；正式 v2 默认继续使用算法 V5，旧算法 V4 仍可选择构建。

## 目录

```text
src/                  当前算法与可构建的旧算法
  reconstruction/     V5 主程序
  legacy/             V4 兼容入口 new_try.cpp
experiments/          单视角、多视角实验（不参与默认构建）
archive/algorithms/   历史日期版本源码
data/samples/         多视角图片与 RGB-D 样例
build/vs/             Visual Studio 工程与筛选器
config/               集中依赖配置及本机覆盖模板
docs/                 使用、构建、版本、迁移与验证文档
scripts/              启动与目录完整性检查工具
models/               本地 ONNX 权重（不入 Git）
runtime/              本地第三方 DLL（不入 Git）
outputs/              生成结果（不入 Git）
```

根目录保留解决方案 `multi-view-3d-reconstruction.sln`。编译输出位于 `build/bin/`，中间文件位于 `build/obj/`，推理缓存位于 `.cache/`。

## 快速开始

环境：Windows、Visual Studio 2022 的 C++ 桌面开发组件、MSVC v143、Windows SDK，以及 x64 freeglut、OpenCV 4.12.0、ONNX Runtime。

1. 按 [构建指南](docs/building.md) 配置依赖，将三种运行时 DLL 放入 `runtime/`。
2. 将兼容的 Depth Anything V2 Large ONNX 放入 `models/depth_anything_v2_large.onnx`。
3. 打开根目录 `.sln`，选择 **Debug | x64** 并生成，按 F5。工程自动将工作目录设为仓库根目录。
4. 当前默认图片是 `data/samples/multiview/` 中的 `f04/b04/l04/r04/t04.png`，结果位于 `outputs/v5/`。

在仓库根目录的 PowerShell 中也可运行：

```powershell
# 在 VS Developer PowerShell 中编译
msbuild .\multi-view-3d-reconstruction.sln /m /p:Configuration=Debug /p:Platform=x64

# 无需 ONNX 的逻辑自检
& .\scripts\run.ps1 --self-test

# 使用默认样例与 ONNX 重建，完成后退出
& .\scripts\run.ps1 --headless

# 不加载 ONNX 的轮廓重建
& .\scripts\run.ps1 --headless --silhouette-only --grid 64 --output-dir outputs/my-check
```

鼠标左键旋转，滚轮缩放；`L` 切换光照，`T` 切换纹理，`R` 重置，`Esc` 退出。更多参数及算法边界见 [使用说明](docs/usage.md)。

## 保留范围

所有原始源码、文档、工程和 108 张样例图片均在版本历史中保留；逐文件位置映射见 [迁移清单](docs/migration-manifest.json)。原 README 与原 V5 说明还保存在 `docs/archive/`，用于核查历史描述。

模型、DLL、生成结果和缓存沿用原仓库的忽略策略，本地移动后保留；旧构建产物与用户配置归档至 `archive/local-v1/`。这些本地大文件不包含在远程源码分支中，新检出时需自行准备依赖。

贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
