# 贡献指南

请先按 [构建指南](docs/building.md) 准备 x64 依赖，再阅读 [使用说明](docs/usage.md) 了解输入约定和算法边界。

当前算法修改放在 `src/reconstruction/`；兼容修复放在 `src/legacy/`；独立算法尝试放在 `experiments/` 并说明输入、入口和验证状态。历史日期版本保留在 `archive/algorithms/`。增加可复用模块时再引入对应头文件目录，避免空目录与重复入口。

请使用相对仓库根目录的资源路径；本机依赖位置写入 `config/dependencies.local.props`。不要提交模型、DLL、构建结果、生成图片或 IDE 缓存。新增外部样例应说明来源和使用条件。

提交前运行相关 Debug/Release 构建、V5 `--self-test`，资源或路径变化另跑低分辨率 headless 样例。算法变更需记录输入、参数、输出指标与限制；在 `docs/validation.md` 中区分新验证与历史记录。目录变化需同步解决方案、工程筛选器、文档和迁移清单。
