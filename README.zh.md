# 权重空间实验室

0.3 版新增独立的 Qwen BF16 权重重建与评估脚本，详见 [Qwen 指南](qwen/README.md) 和 [结果报告](docs/REPORT.md)。两个编辑张量的哈希及两道题目的试验已验证；完整 MATH-500 评估尚未完成，新的语言模型编辑发现流程尚未实现。


0.2 版新增真实研究复现流程：下载固定版本的 MinAtar Breakout 提示词与权重，校验哈希，从原始回复精确重建权重，执行 1,000 局评估，再通过无工具的 GPT-6 Astra 单次调用生成新网络并使用相同条件评估。两种宿主均可调用同一脚本；自动生成阶段使用现有 Codex 登录。详见 `skills/weight-space-lab/references/reproduction.md`。这不是 Qwen 数学基准的复现；原始评估随机种子未公开，因此比较的是统计结果。

2026 年 11 月 4 日实测：发布网络本地重放为 502.9 ± 7.7 标准误，新的单次 Astra 响应为 511.5 ± 5.5 标准误，均为 1,000 局；文章结果为 517.4 ± 7.4。新网络未经过重试或评估反馈调优。协议、哈希与审核记录见 `RESULTS.md`。

适用于 Codex 和 DeepSeek Harness 的本地实验插件，两者共用同一个技能和 Python 工具。代理提出假设，工具负责检查权重、保存独立候选检查点、记录哈希与改动数量，以及比较评估结果。它不能访问或修改宿主助手的权重。

需要 uv，或安装 NumPy 2.5.4、SafeTensors 0.8.0 的 Python 3.11+ 环境。在插件目录运行：

```sh
uv run --script skills/weight-space-lab/scripts/weight_lab.py demo ./demo-run
```

演示使用预先编写的符号分类器修复，不代表代理自主发现能力。支持单个 NPZ 或 SafeTensors 文件、F16/F32/F64 编辑、逐元素修改和指定矩阵列的方向投影移除。原始文件保留，输出必须是新目录。内置评估仅适用于全连接分类网络；Transformer 需要现有的专用评估程序。暂不支持 BF16、量化权重、自动处理分片、下载模型、训练或部署。

Codex：通过本地插件市场安装包含 `.codex-plugin/plugin.json` 的目录。DSH：在 Creator 会话中用 `plugin_manager` 的 `install_bundle` 安装本目录，或运行 `dsh plugin --profile desktop add /absolute/path/to/weight-space-lab`。按安装结果决定是否重启，不要手动修改 profile。卸载使用 `remove_bundle`，目标为 `dsh-weight-space-lab`。

详细格式见 `skills/weight-space-lab/references/cli.md`。本项目是受权重空间编程研究启发的独立工具，不复现文章中的 Qwen 基准结果，也不保证能力提升。英文 README 包含来源、开发和打包说明。
