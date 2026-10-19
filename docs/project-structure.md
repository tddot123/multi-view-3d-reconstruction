# 项目结构与迁移

正式 v2 按职责划分：`src/reconstruction` 为当前 V5，`src/legacy` 为可选择构建的 V4；`experiments/single_view` 收纳四个 front 实验，`experiments/multiview` 收纳 car_box、copy、test、try_depth_range；`archive/algorithms` 收纳四个按日期命名的历史版本。保留原文件名、注释和算法实现，只修改迁移所需的文件路径及说明链接。

`test.cpp` 是原多视角算法实验，不是自动化测试，因此归入 experiments。内置逻辑自检仍在 V5 源码中，通过 `--self-test` 执行。

原 `openGL_enhance/input` → `data/samples/multiview`；`test_img` → `data/samples/rgbd`；工程和筛选器 → `build/vs`；V5 说明 → `docs/usage.md`。旧 `six/` 视角路径统一指向已有多视角样例，单视角实验的裸 `color4.jpg/depth4.png` 路径指向 RGB-D 样例。`car_box.cpp` 原本引用不存在的 `f0.png`，此次保留该选择并记录为历史实验已有问题；未宣称全部实验可直接运行。

模型移至 `models/`，DLL 移至 `runtime/`，结果移至 `outputs/`，V5 深度缓存移至 `.cache/v5-depth-cache`。由于缓存键包含图片/模型绝对路径，迁移后的首次 ONNX 推理可能重新计算，旧缓存仍保留。

原目录中剩余的用户配置和构建文件，以及根目录旧 Debug/x64 产物，移动到本地忽略目录 `archive/local-v1/`。根目录 `.vs/` 保持本地 IDE 管理。未删除这些内容，也未将缓存或第三方大文件提交到 Git。

所有版本控制文件的位置映射及内容摘要记录在 `migration-manifest.json`。`v1_sha256` 对应迁移开始时磁盘原始字节；`v1_git_sha256` 和 `v2_sha256` 对文本按 LF 规范化，以兼容 Git 的 CRLF 转换。运行 `python scripts/verify_layout.py` 可验证目标文件完整性、原始 v1 Git 内容和工程引用。

原始 README 与 V5 说明另存于 `docs/archive/`，内容按历史快照保留，其中旧路径和旧默认值不适用于正式 v2。当前说明已核对源码，将默认图片更正为 `f04/b04/l04/r04/t04.png`。

本次不拆解算法内部模块；以目录、配置和文档整理为范围，降低改动对重建行为的影响。
