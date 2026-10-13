# Multi-View 3D Reconstruction and Visualization

**基于深度估计的多视角三维重建与可视化系统**

本项目使用 C++、OpenCV、ONNX Runtime 和 OpenGL/freeglut，从前、后、左、右、顶等视角图像估计深度，进行体素重建，并提供三维交互式可视化。

GitHub 仓库：`tddot123/multi-view-3d-reconstruction`。

## 项目内容

- `multi-view-3d-reconstruction.sln`：Visual Studio 解决方案。
- `openGL_enhance/multi-view-3d-reconstruction.vcxproj`：C++ 工程。
- `openGL_enhance/new_try.cpp`：当前启用的程序入口，包含多视角深度估计与体素重建流程。
- `openGL_enhance/v*.cpp` 及其他实验源码：保留的历史实验版本。
- `openGL_enhance/input/`：多视角输入样例。
- `openGL_enhance/test_img/`：图像与深度测试样例。

源码目录保留原名 `openGL_enhance`，解决方案、工程显示名称及仓库统一采用新名称。

## 开发环境

- Windows、Visual Studio 2022，安装 C++ 桌面开发组件。
- MSVC v143、Windows SDK。
- OpenGL、freeglut、OpenCV 4.12.0、ONNX Runtime。
- Depth Anything V2 Large 的 ONNX 模型。

现有工程在 Debug 配置中使用以下本机依赖目录；换机器后请在工程属性中调整包含目录、库目录和链接依赖：

```text
D:\freeglut\include
D:\freeglut\lib\x64
D:\opencv\build\include
D:\opencv\build\x64\vc16\lib
D:\onnx\build\native\include
D:\onnx\runtimes\win-x64\native
```

建议使用 `Debug | x64`。Release 及 Win32 配置尚需按实际依赖补齐和验证。

## 编译与运行

1. 打开 `multi-view-3d-reconstruction.sln`，选择 `Debug | x64`。
2. 安装以上依赖，确认包含目录、库目录和运行时 DLL 可用。
3. 将兼容的模型文件放置在 `openGL_enhance/model/depth_anything_v2_large.onnx`。
4. 将运行工作目录设置为工程目录 `openGL_enhance`，确保能够读取 `input/` 和 `model/`。
5. 创建 `openGL_enhance/output/` 目录，用于保存生成的深度图和调试结果。
6. 编译并启动程序。当前入口读取 `input/front.png`、`back.png`、`left.png`、`right.png` 和 `top.png`。

这是实验开发项目；仓库保留现有源码与配置，不代表所有历史实验版本均已验证可运行。

## 版本控制范围

源码、工程配置和输入样例纳入 Git。Visual Studio 缓存、编译产物、第三方 DLL/库、ONNX 模型权重及 `output/` 生成结果由 `.gitignore` 排除，需要在本地另行准备。重命名与上传不会删除已有本地依赖或输出。

## 日常提交

```powershell
git status
git add .
git commit -m "描述本次修改"
git push
```
