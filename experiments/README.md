# 实验源码

- `single_view/`：front_mesh、front_mesh_colordepth、front_pixel、front_success。
- `multiview/`：car_box、copy、test、try_depth_range。

保留原文件和注释状态，只调整迁移后资源路径。它们不参与当前构建，也不是自动化测试。部分实验已有缺失输入或手工配置要求，例如 car_box 的 `f0.png` 不在样例中。恢复实验时单独选择入口、检查输入和依赖，再执行验证。
