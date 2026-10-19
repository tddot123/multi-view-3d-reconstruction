# 构建与依赖

## 支持配置

Visual Studio 2022、MSVC v143、C++17、Windows SDK，平台固定为 x64。原 Win32 配置引用 x64 库且会主动报错，正式 v2 从解决方案中移除这些无效配置。Debug 与 Release 分别链接带 `d` 后缀和不带后缀的 OpenCV 库。

## 依赖位置

`config/dependencies.props` 集中管理路径，默认沿用此机器的 `D:\freeglut`、`D:\opencv\build`、`D:\onnx`。换机器无需修改工程文件：

```powershell
Copy-Item config/dependencies.local.props.example config/dependencies.local.props
```

编辑被 Git 忽略的 `dependencies.local.props`，设置 `FreeglutRoot`、`OpenCVRoot`、`OnnxRoot`、`OpenCVVersion`；特殊 OpenCV 布局可另设 `OpenCVLibraryDir`。MSBuild 命令行 `/p:FreeglutRoot=...` 等也可覆盖。文件夹视图使用的根目录 `CppProperties.json` 保留原本机 IntelliSense 路径，其他机器需同步调整；实际编译使用上述 MSBuild 配置。

所需目录关系：freeglut 的 `include`、`lib/x64`；OpenCV 的 `include`、`x64/vc16/lib`；ONNX 的 `build/native/include`、`runtimes/win-x64/native`。

将以下 DLL 放入 `runtime/`（按实际 OpenCV 版本调整文件名）：

- `freeglut.dll`
- `opencv_world4120d.dll`（Debug）及 `opencv_world4120.dll`（Release）
- `onnxruntime.dll`

构建后自动复制所选配置需要的 DLL 至程序目录，缺少 DLL 时部署步骤会报错。自定义运行时目录可覆盖 `RuntimeDir`，请保留末尾路径分隔符。ONNX 模型放入 `models/depth_anything_v2_large.onnx`，或运行 V5 时传 `--model`。仓库不分发上述第三方二进制和模型。

## 编译与启动

在 Visual Studio Developer PowerShell 中，从仓库根目录运行：

```powershell
msbuild .\multi-view-3d-reconstruction.sln /m /p:Configuration=Debug /p:Platform=x64
msbuild .\multi-view-3d-reconstruction.sln /m /p:Configuration=Release /p:Platform=x64

# 构建兼容旧算法（不是切换正式发行分支）
msbuild .\multi-view-3d-reconstruction.sln /m /p:Configuration=Debug /p:Platform=x64 /p:ReconstructionVersion=v4
```

可执行文件路径为 `build/bin/x64/<Debug|Release>/<v4|v5>/multi-view-3d-reconstruction.exe`，中间目录同样按配置和算法隔离，避免互相覆盖。

VS F5 的工作目录已在受版本管理的工程中设为仓库根目录，无需依赖 `.vcxproj.user`。命令行直接运行 exe 时也应从仓库根目录执行；或从任意目录调用 `scripts/run.ps1`，该脚本会定位仓库根目录并转发 V5 参数。脚本默认运行 Debug V5；Release/V4 请直接运行对应 exe。V4 沿用旧交互流程，不支持 V5 CLI 参数。

历史算法和实验源码仅作为工程浏览项保留，不参与编译。恢复单个实验需检查注释状态、入口和依赖，不能简单全部加入编译，否则可能出现多个 `main`。
