# 正式 v2 本次验证

本次目录迁移在 Windows 本机、VS 2022/MSVC v143、OpenCV 4.12.0、已有 x64 freeglut 和 ONNX Runtime 环境中执行以下验证。这里记录本次实际执行结果；`usage.md` 中其他算法历史验证描述保留为历史记录。

## 构建与运行

- 算法 V5：Debug x64 与 Release x64 构建成功，DLL 自动复制成功。
- 算法 V4：Debug x64 构建成功，独立构建目录生成；未执行旧版交互重建。
- Debug V5 `--self-test` 成功；通过 `scripts/run.ps1 --self-test` 再次确认启动脚本定位仓库及参数转发有效。
- 默认 04 组五视图，Debug V5 `--headless --silhouette-only --grid 64 --output-dir outputs/validation-v2/silhouette` 成功；32892 个顶点、65780 个三角形、单连通体、开放边/非流形边/朝向不一致边均为 0。
- 默认 04 组五视图，Release V5 `--headless --grid 64 --input-size 140 --output-dir outputs/validation-v2/onnx` 成功；实际读取迁移后 ONNX 模型并执行五视图推理，32522 个顶点、65040 个三角形、单连通体、开放边/非流形边/朝向不一致边均为 0。
- 使用 Release V5 `--preview-ply outputs/validation-v2/onnx/surface.ply --snapshot --output-dir outputs/validation-v2/preview` 成功打开迁移后模型和纹理资源，生成 OpenGL `preview.png` 并退出。

上述结果验证资源路径、模型加载、推理、网格导出与查看链路可运行，不构成几何精度提升的结论。默认 256 网格、518 推理尺寸以及所有历史实验未重新全面运行。生成结果保留在本地 `outputs/validation-v2/`，沿用忽略策略不提交 Git。

## 内容完整性

正式 v1 快照共 129 个受控文件；迁移清单逐一映射至正式 v2，108 张样例图片按原字节保留。14 个 C++ 文件经逐文件对比，变动仅为约定的资源路径和说明链接替换，无算法逻辑改动。原 README 和 V5 说明副本与正式 v1 内容一致（忽略 Git 换行转换）。

`python scripts/verify_layout.py` 检查上述清单、摘要、原文档副本、工程源码引用、解决方案路径和当前文档链接。该脚本使用 Python 标准库和 Git，需保留正常仓库历史；仅下载 ZIP 或历史不完整的浅克隆不能校验 v1 提交。

`git diff --check` 用于检查提交中的空白问题。模型、运行时 DLL、结果与旧构建文件以本地移动方式保留，未纳入 Git；远程分支保存可复现的源码与配置，第三方依赖仍须按构建指南准备。
