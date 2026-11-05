# AgentPrune 精简迁移

保留本地 AgentPrune 已实现的两套对齐入口：

- `experiments.run_mmlu_attrprune_aligned`
- `experiments.run_gsm8k_attrprune_seed0`

统一启动方式见上一级 README。直接运行时，把本目录和 AgentAttrPrune 项目根目录放入 `PYTHONPATH`；也可通过 `BASELINES_ATTRPRUNE_ROOT` 指定后者。两个入口均支持 `--dry-run`、`--stage prepare/train/eval/all`、`--resume`、`--reference-run` 和 `--run-dir`。`prepare` 只校验并保存元信息，无需 GPU。

## 保留的算法

`AgentPrune/graph/graph.py`、`node.py` 和 registry 来自原项目。使用 sigmoid/Bernoulli 采样通信边，空间边构图时排除环；跨轮共享边权，两轮均执行，即使某个节点没有入边。训练后冻结参数和掩码，评测仍按原方法采样保留边。

| 配置 | MMLU | GSM8K |
|---|---|---|
| worker / 轮数 | 5 / 2 | 4 / 2 |
| 校准题池 | AgentAttrPrune 原始 seed0 dev40 | AgentAttrPrune 原始 seed0 train40 |
| 训练预算 | 160 rollout，10 次 Adam 更新；逻辑 batch=16 | 40 道题，每题 10 rollout，40 次 Adam 更新；逻辑 batch=1 道题 |
| 损失 | `mean(-log_probability * correctness)` | 同左，加原始 raw-logit 核范数与 Frobenius hinge 正则，delta=0.1 |
| 学习率 | 0.1 | 0.1 |
| 剪枝 | 第 5、10 次更新后，空间和时间边各剪去当时剩余边的 25% | 最后一次更新后，空间和时间边各一次剪去 50% |
| 评测集 | val 全部 1531 题 | test 全部 1319 题 |

生成微批次上限和评测并发都是 **4**，但不改上表中算法自身的逻辑训练 batch。prompt、匿名 reviewer ID、temperature=0、max_new_tokens=2048、enable_thinking=False、HFChat 后端均来自当前 AgentAttrPrune。答案解析仍保留原 AgentPrune 的 MMLU/GSM8K 规则，因此不能把这些规则也当作与 AgentAttrPrune 完全相同。数据与题目顺序由共享 alignment 模块校验，不使用原 AgentPrune 的数据加载代码；参考 run_config 指定的物理数据路径仍会被读取，当前 MMLU 参考配置的数据位于 `../AgentPrune/dataset/MMLU/data`。因此不能直接删除该数据目录，迁移该目录时还需处理参考配置和冻结校验。

## 迁移范围与依赖

只保留图算法、两套 adapter/runner、policy 导出、答案解析和 GSM8K 正则。去掉 AutoGen、工具调用、旧提示词、云模型后端、旧数据加载器和其他任务。`dataset/` 中只有答案解析代码，没有数据集。`experiments/regularization.py` 的函数体从原 GSM8K 入口原样提取。MMLU 的 `prepare` 阶段是与 GSM8K 对齐的元信息准备功能，不改变训练。

`assets/*_protocol.json` 仅保存已有实验的训练协议参数，不包含学习到的权重，也不依赖历史结果目录。来源、源文件 SHA256 和本地 Git revision 记录在 `PROVENANCE.json`；本地工作区中已存在的修改也通过文件 SHA256 标识。共享模型、数据和冻结参考 traces 仍位于 AgentAttrPrune 及其配置指定的位置，迁移没有复制大文件。若参考数据、prompt 或参考 traces 改变，现有严格校验会拒绝运行，需使用有效的 `--reference-run`。

依赖：AgentAttrPrune 的 Python 依赖，加 `class-registry==2.1.2`。测试另需 pytest。结果包括 `manifest.json`、`alignment.json`、`checkpoint.json`、训练/评测 traces、pruning policy 和 `summary.json`；恢复时继续使用同一个 `--run-dir` 并传 `--resume`。新代码的 fingerprint 与旧目录中的运行不同，因此不能直接接着旧目录 checkpoint 续跑。

两组测试由已有 aligned 测试迁入，用假的模型验证 prompt、图采样/梯度、真实 HFChat 微批次分组和断点恢复；图构造测试改用本包保留的 aligned 注册节点，避免引入旧模型后端。

原始仓库：[yanweiyue/AgentPrune](https://github.com/yanweiyue/AgentPrune)。本地源目录没有顶层 LICENSE 文件；这里保留来源记录，没有新增或替换其许可声明。
