# V5 使用说明

当前程序入口是 `v5_confidence.cpp`，旧的 `new_try.cpp` 保持原样。工程默认编译 V5，使用 **Debug | x64**。

## 本版解决的问题

- `View` 同时持有 RGB、mask 和浮点深度；裁剪、旋转、镜像、尺寸对齐同步完成。沿用旧项目的坐标约定：left/back 水平翻转，top 逆时针旋转 90 度。
- 实际检查本地 ONNX 图：输入为动态 NCHW float，`pixel_values` 直接进入 patch-embedding Conv，图中没有输入归一化。因此本版在模型外执行 RGB 转换及 ImageNet mean/std 归一化。默认推理长边约 518，宽高取 14 的倍数并保持比例；长边限制是面向本机 CPU 的折中，并非官方默认的短边缩放策略。
- 深度始终使用 float。mask 内部的 2%/98% 分位数用于归一化，减少离群值拉长值域；不再使用 front/back 的 P95 总和强行放大。8 位 PNG 仅用于查看。
- 投影单元区分可靠背景、前景和未知边界；前景深度不可信时保留轮廓，不计负证据。一个有效像素也可以形成单元，不再要求至少两个样本。
- 候选空间由所有真实视角的轮廓共同约束，至少两个真实轮廓支持；不再以 front 的深度体积作为不可扩张的初始空间。
- 按样本数量、覆盖率、深度离散程度及梯度调整置信度，实际用于支持和拒绝权重。固定视角累加顺序保证结果不依赖传入的视角顺序。
- 缺少 back/right 时复用相对视角的对齐结果，作为低权重镜像先验；不参与真实视角数量、硬轮廓裁剪或独立支持计票，也不重复执行 ONNX。
- 相对深度锚定到轮廓已经确定的车身表面，只在受限的表面带内细化，不再把单张图的 min/max 拉满整个三维轴向。体素诊断与连续表面都保留这种约束。
- 默认使用原始高分辨率 mask 的连续有符号距离场，融合真实视角的轮廓，再加入小幅、平滑的深度细化。它避免了先取整到体素再生成表面产生的车顶、风挡和发动机盖台阶。
- 用 Marching Tetrahedra 提取共享顶点的三角网格，依据每个四面体的内外顶点确定面朝向，再做 Taubin 平滑并重算法线；检查开放边、非流形边、共享边朝向和连通体数量。采用自动 mask 的照片只清理极小的悬浮数值碎片；提供人工 mask/alpha 时保留独立部件。
- 表面交点按原始网格边缓存，不再按近零插值比例合并到节点；使用固定端点顺序和双精度插值，将交点限制在边的内部（比例距离端点至少 0.0001）。这样避免浮点舍入和极小三角面丢弃造成的数值裂缝；插值比例最多调整 0.0001。闭合性与朝向检查继续保留，在纹理着色前执行。
- 纹理从原始对齐图片投影，按连续法线、mask、实际网格的投影深度可见性和观察方向融合，避免各个阶梯面交替取 front/top 的颜色。顶视图被车身遮挡的底部不再借用车顶图片；无观测区域保持暗色。向轮廓外延伸材质颜色以减少白边。OpenGL 显示高分辨率纹理，PLY 保存顶点色和法线。
- 原始浮点深度缓存到 `.build/v5-depth-cache/`。缓存检查输入、模型的绝对路径、大小、修改时间及推理尺寸/预处理版本，改变输入或模型时重新推理；mask 或网格参数修改无需重复推理。`--no-cache` 可强制重算。

## 输入与 mask

在 `v5_confidence.cpp` 顶部的 `Config` 中修改 `frontFile`、`backFile`、`leftFile`、`rightFile`、`topFile`，即可直接选择 `input` 文件夹里的另一组图片。例如分别填写 `f03.png`、`b03.png`、`l03.png`、`r03.png`、`t03.png`。修改源码后需要重新生成项目；程序启动时打印实际读取路径。

默认文件名仍为 `front.png`、`back.png`、`left.png`、`right.png`、`top.png`。相对文件名基于 `inputDir`，也可填写绝对路径。front/left/top 必须存在；back/right 可缺失，或填写空字符串 `""` 使用镜像先验。`--input-dir` 继续用于切换输入目录，它不会重置代码中的文件名。VS 若保留了上一组的调试命令参数，运行前要清除或更新目录参数。

可在原图所在目录提供同名加 `_mask` 的外部灰度 mask，例如 `f03.png` 对应 `f03_mask.png`，默认 `front.png` 对应 `front_mask.png`：白色为物体，黑色为背景，尺寸必须与对应的原始 RGB 一致。外部 mask 不执行自动形态学或组件清理。带有效透明通道的 PNG 也可以直接使用 alpha。

无外部 mask 时，先用白背景 HSV 阈值找出彩色/较暗的主体，再提取最大主体的外部轮廓并填充内部。这样当前汽车的玻璃、镀铬、高光不会被误当成穿透车身的孔洞，也能排除大部分水印和旁边的噪点。自动方式有意把封闭内部区域作为材质；**真实孔洞、独立小部件、纯白/纯灰物体请提供外部 mask 或透明 alpha**，这些输入会绕过自动填充与组件清理。自动轮廓不能保证适用于所有照片，请先查看 `*_mask.png`。

## VS 运行

打开根目录的 `multi-view-3d-reconstruction.sln`，选择 Debug | x64，重新生成并按 F5。原有调试工作目录与 DLL 部署设置继续有效。

鼠标左键旋转，滚轮缩放；`L` 切换光照，`T` 切换纹理/纯几何，`R` 重置视角，`Esc` 退出。切换显示不改变导出的几何。

以下 PowerShell 命令在 `D:\openGL_enhance\openGL_enhance` 目录执行：

```powershell
# 不加载模型的逻辑自检
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --self-test

# 完整建模及导出，结束后退出，不显示 OpenGL 窗口
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --headless

# 验证 OpenGL 渲染，将初始视角保存为 preview.png 并自动退出
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --snapshot

# 初始视角 + 前后左右顶部 + 纯几何，检查整车而不是只看一个角度
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --snapshot-views

# 重新打开已经导出的 V5 模型，跳过 ONNX 和重建
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --preview-ply output/v5/surface.ply

# 仅用真实轮廓构建参照结果，不加载 ONNX
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --headless --silhouette-only --output-dir output/v5-silhouette

# 自定义输入集、输出目录和分辨率
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --headless --input-dir input/my-object --output-dir output/my-object --grid 256 --input-size 518

# 手工校正顶视图深度范围；这是相对深度参数，不是米制标定
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --headless --top-scale 0.8 --top-offset 0.1 --output-dir output/v5-top-adjusted

# 对照无深度细化的完整轮廓模型
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --snapshot-views --depth-relief 0 --output-dir output/v5-no-relief

# 复用已有网格调整观察角度，不进行 ONNX 或重建
& '..\x64\Debug\multi-view-3d-reconstruction.exe' --preview-ply output/v5/surface.ply --snapshot --yaw -55 --pitch 25 --output-dir output/v5-left-angle
```

参数范围：`--grid` 为 4–384，`--input-size` 为 140–1036，`--threads` 为 1–16（默认 2）。`--depth-scale` / `--depth-offset` 对全部视角生效；`--front-scale` / `--front-offset` 等可分别调整各方向，校正后裁剪到 [0,1]。

`--depth-relief` 为 0–6（默认 1.5），它限制轮廓附近的细化幅度，连续距离场只使用该上限的小比例，以保护完整车身。`--smooth-iterations` 为 0–10，默认 3。`--surface-margin` 为 0–2 个体素，默认 0.35，用于处理轮廓边界的轻微误差。

`--front-photo-pitch` / `--back-photo-pitch` 可在 ±30 度内试调纹理投影俯角，默认都为 0：现有样例使用零俯角能更好保留车灯和车牌的位置。这是近似纹理投影参数，不能代替相机标定。`--yaw` / `--pitch` 只改变初始观察角度。

`--voxel-surface` 回到体素密度场提取表面，便于比较连续轮廓与整数网格。`--legacy-depth` 可比较原来的绝对值域映射；这个模式可能再次破坏车身完整性，仅用于对照。`--reject-strength` 主要控制体素雕刻诊断/旧映射模式，默认连续表面优先遵守轮廓距离场和有限深度细化。

本地 ONNX 已确认需要模型外归一化；若换成内置预处理或 metric head 的另一份 ONNX，需核对并修改 `DepthModel`，不能直接认为兼容。

## 输出

默认保存至 `output/v5/`，与旧版结果分开：

- `*_aligned.png`、`*_mask.png`、`*_depth.png`：对齐颜色、二值轮廓、深度预览。
- `*_states.png`：白色为前景，黑色为可靠背景，灰色为未知边界。
- `*_confidence.png`：越亮，深度约束置信度越高。
- `surface.ply`：焊接、平滑后的三角网格，含 RGB 和法线；坐标以模型中心为原点，单位为一个体素，可导入 MeshLab/Blender。
- `report.txt`：网格尺寸、体素诊断、顶点/三角形数量、连通体、开放/非流形边、朝向不一致的共享边，以及真实视角下的体素与实际网格投影覆盖率/轮廓 IoU。
- `preview.png`：仅在 `--snapshot` 模式生成的实际 OpenGL 渲染图。
- `preview_front/back/left/right/top.png`、`preview_geometry.png`：`--snapshot-views` 生成的多角度和纯几何检查图。

PLY 使用共享顶点，支持连续法线和自动平滑；表面提取是 Marching Tetrahedra。OpenGL 纹理来自同目录的对齐图片，PLY 本身保存顶点色而非 UV 材质；移动文件后仅有 PLY 也能显示顶点色。每次运行会覆盖当前输出目录，可以指定不同目录保存对照。

## 验证与边界

`--self-test` 检查同步变换、稳健值域、单像素单元、未知区域、视角顺序、镜像投票、受保护的轮廓内部、遮挡纹理，以及平滑立方体的连通性/闭合性和反向面的识别。端到端验证另用已知尺寸立方体与人工 mask 的带孔物体，独立解析 PLY 检查索引、色值、法线朝向和边共享数量。当前汽车的 PLY 还使用独立 NumPy/SciPy 解析器检查单一连通体、所有共享边的相反方向及正有向体积，检查结果保存在 `mesh_geometry_check.json`。

自检还覆盖近零正值与恰好为零的节点值，在较大的浮点坐标下验证网格不会塌缩或出现开放边。对于输入 `f2/b2/l2/r2/t2.png` 曾触发的 `Extracted surface is not watertight`，应重新生成新版后运行；无需重装 OpenCV 的可选并行插件。

本版仍使用轴对齐、近似正交投影，保留裁剪后按 W/H/D 对齐的旧路线；不包含相机标定、SfM、多视图匹配、自动跨视图尺度优化或米制 TSDF。相对深度归一化与手动范围修正不能替代这些方法。当前仅有五个方向，没有 bottom 图。真实照片的透视误差和不一致拍摄姿态仍可能影响最终形状。

建议先比较真实图片的 V5 与 silhouette-only 结果，再用有已知几何的立方体、带孔物体和细杆判断是否误删。逻辑测试及成功运行说明工程和算法路径可用，不能证明所有输入的几何精度都优于旧版。

## 回到旧版

无需修改旧源码。在工程的 `<ReconstructionVersion>` 中将默认值 `v5` 改为 `v4`，重新生成即可恢复 `new_try.cpp`。也可使用 MSBuild 参数 `/p:ReconstructionVersion=v4` 编译旧版。`v4` / `v5` 分别使用独立中间目录，两个入口不会同时编译。
