# AgentAttrPrune 对齐 baselines

这里整理了本地 `AgentPrune`、`AgentDropout`、`SafeSieve` 中**已经实现的对齐版本**，用于从自己的实验目录运行 MMLU / GSM8K 对比。本目录位于 `AgentAttrPrune/baselines`，随当前仓库一起版本管理；原先整理的同级 `Project/baselines` 保持原样，本仓库不从该副本导入代码。

只迁移图算法、训练/评测入口、prompt adapter、结果/断点处理及少量协议资产。没有复制模型、数据集、历史实验结果，也没有重新实现三种剪枝算法。三者的核心 `graph.py` / `node.py` 沿用源实现；去除了与本次实验无关的云 API、工具调用、其他 benchmark 和旧入口依赖。

## 直接运行

使用已有的 `agentprune` Python 环境即可。本机验证所用解释器：

```bash
conda activate agentprune
# 以下命令均可从 AgentAttrPrune 项目根目录执行。
python baselines/run.py all mmlu --dry-run
python baselines/run.py all gsm8k --dry-run
```

`--dry-run` 实际检查冻结题池、prompt、源文件 hash、协议参数及必要资产，不加载模型、不写结果。只看将执行的命令可用 `--print-command`。

```bash
# 单个方法：训练并评测；0,1 替换为实际选用的 GPU。
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py agentprune mmlu --stage all
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py agentdropout gsm8k --stage all
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py safesieve gsm8k --stage all

# 同一 benchmark 的三个方法依次运行，某个方法失败即停止。
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py all mmlu --stage all
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py all gsm8k --stage all

# 结果写到自己的目录；相对路径按启动命令时的工作目录解释。
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py all mmlu \
  --output-root ./results/baselines --run-name comparison_v1

# 单个方法可精确指定目录，继续该目录自己的 checkpoint。
CUDA_VISIBLE_DEVICES=0,1 python baselines/run.py agentprune gsm8k \
  --run-dir ./results/baselines/ap_gsm8k --resume
```

也可在 `baselines` 目录运行 `python run.py ...`，或从任意目录使用 `run.py` 的绝对路径。统一入口为每个方法启动独立 Python 进程，并设置自己的模块路径；不要在同一进程中混合导入三套同名 `experiments` 包。

`--stage prepare` 只保存元信息；`train` 训练；`eval` 读取本次运行已训练的策略评测；`all` 训练后评测。默认不覆盖已有运行，继续同一个结果目录时传 `--resume`。AgentPrune、SafeSieve 保留训练断点恢复；AgentDropout 对齐入口只能复用完整训练结果和续评测，训练中断且没有完整 checkpoint 时需另开目录。迁移后的代码路径/hash 已改变，不能直接接着原三个项目的旧 checkpoint 运行。

## 对齐了哪些设置

| 项目 | MMLU | GSM8K |
| --- | --- | --- |
| 默认参考实验 | `AgentAttrPrune/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware` | `AgentAttrPrune/results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware` |
| 校准题池 | 原 seed0 dev40 | 原 seed0 train40 |
| 评测集 | val，1531 题 | test，1319 题 |
| workers / rounds | 5 / 2，另有 reviewer | 4 / 2，另有 reviewer |
| prompt | 当前 `MMLUPromptSet` worker/reviewer builders | 当前 `GSM8KPromptSet` worker/reviewer builders |
| 解码 | temperature=0，max_new_tokens=2048，thinking=false | 同左 |
| 评测 batch / 生成微批次上限 | **4 / 4** | **4 / 4** |
| 模型、worker 顺序、匿名 reviewer ID | 从冻结参考配置读取 | 从冻结参考配置读取 |

模型默认与当前参考实验一致，即本机 Qwen3-8B。实验 seed=0；当前参考数据的 shuffle_seed=888，题池由保存的 selection manifest / traces 验证，不能把“seed0”理解为原始文件的前 40 行。尤其 GSM8K 默认采用原始 AgentAttrPrune train40，**不是**后来的 `train40_baseline_aligned_correctness_aware` 原文件顺序 cohort，也不是重采样 cohort。40 题包括最终答错的题，不按正确性筛选。

共享 alignment 会逐条核对当前 prompt 与参考 traces，校验题目、答案、题序和数据 hash；若当前 prompt 已改动而参考 traces 未更新，会明确报错。`--reference-run` 可指定满足同一契约的参考运行，不是任意历史结果目录。

batch=4 是推理层设置，保留原方法的逻辑训练批次与训练预算：

| 方法 | MMLU 训练 / 剪枝 | GSM8K 训练 / 剪枝 |
| --- | --- | --- |
| AgentPrune | 10 次更新 × 16 rollout；第 5、10 次更新按原规则各剪 25% 空间/时间边 | 40 题 × 每题 10 rollout；40 次更新后一次剪 50% 空间/时间边 |
| AgentDropout | 节点阶段 10×16，边阶段 10×16；边阶段第 5、10 次更新按原规则剪 25% | 节点阶段 2×20，边阶段 4×10；边阶段第 2、4 次更新按原规则剪 10% |
| SafeSieve | 循环 dev40 产生 285 draws，逻辑 batch=16；第 10、15 批满足阈值条件时剪枝 | train40 运行一次，逻辑 batch=4；第 10 批满足阈值条件时剪枝 |

不同方法训练预算、剪枝比例/条件、图采样、原答案解析和奖励计算仍各自保留。原目录占位文档要求的统一评分器、统一 evaluation schema 和嵌入当前 run 的输出路径尚未接入；当前实现提供各方法独立的对齐实验入口及独立结果目录。相同 prompt 指消息模板和身份契约相同；实际收到哪些上游消息由各自图策略决定。它们不是完全相同训练成本或完全相同评分器的实验。

AgentPrune / SafeSieve 直接复用 AgentAttrPrune HFChat；AgentDropout 保留现有本地队列/副本后端，同样 greedy decoding，默认两个完整模型副本，每个副本每次生成至多 4 条。可传 `--replicas 1` 使用一个副本；每个副本需在单张 GPU 放得下模型。实际生成批次可能因图依赖、队列或最后不足四条而小于 4；共享 HFChat 的 OOM 回退也可能拆分批次。

## 最小依赖与目录

```text
baselines/
  run.py                  # 统一入口；默认结果在 results/<method>/<benchmark>/<run-name>/
  requirements.txt        # 当前方法依赖之外的 class-registry
  agentprune/             # 图/策略梯度、原正则、aligned runner、协议 JSON
  agentdropout/           # 节点/边 dropout、aligned runner、本地推理队列
  safesieve/              # 统计/阈值/剪枝、aligned runner、角色相似度缓存
  tests/                  # 统一入口的进程隔离与路径测试
```

各目录的 `METHOD.md` 说明具体算法和迁移边界；`PROVENANCE.json` / `SOURCE_MANIFEST.json` 记录本地源文件、Git revision 和 SHA-256。没有把源项目当前未提交的对齐改动误当成上游仓库的原始版本。

新建环境时先安装当前方法依赖，再补 registry：

```bash
# 从 AgentAttrPrune 根目录执行；已有可用环境不必重装。
python -m pip install -e .
python -m pip install -r baselines/requirements.txt
```

共享代码默认使用本仓库根目录（`baselines/..`）；也可用 `--attrprune-root /path/to/AgentAttrPrune` 或 `BASELINES_ATTRPRUNE_ROOT`。除了安装的包，还需要该项目的 `experiments_attrprune` 源码和冻结参考目录。完整数据/模型路径由参考配置指定：当前 GSM8K 数据在 `AgentAttrPrune/data/GSM8K`，**当前 MMLU 参考数据仍在 `AgentPrune/dataset/MMLU/data`**。因此本包不依赖三个旧项目的 Python 实现或训练结果，但还不能删除该 MMLU 数据目录；没有为迁移复制数据或重写冻结参考文件中的绝对路径。搬到另一台机器时也必须准备有效参考实验及其数据/模型路径。

SafeSieve 必需的角色相似度资产已收进 `safesieve/assets`，不会再调用远程评分模型。MMLU 继承已有实现的限制：缓存来自原日志的两位小数矩阵，无法声明与原运行内部的全精度矩阵逐位一致。GSM8K 使用已有的对齐角色描述缓存。AgentPrune 的协议 JSON 只含参数，不复用历史学习到的策略。

## 验证与输出

首次精简迁移时 **79 项测试通过**（AgentPrune 17、AgentDropout 28、SafeSieve 30、统一入口 4），六个正式入口的 dry-run 均通过，同一 benchmark 的三种方法得到一致的 alignment fingerprint。这些检查包含 CPU / mock 模型训练与恢复，没有启动 GPU 训练或完整评测。随后纳入当前仓库时只调整路径发现、版本管理和文档，重新验证统一入口测试与六个 dry-run；没有重复整套算法测试。重跑检查：

```bash
# 从 baselines 目录执行；测试环境需要 pytest。
python -m pytest tests -q
for method in agentprune agentdropout safesieve; do
  (cd "$method" && python -m pytest tests -q) || break
done
python run.py all mmlu --dry-run
python run.py all gsm8k --dry-run
```

各方法独立运行测试，避免同名模块冲突。共享 prompt/数据契约的测试继续位于 AgentAttrPrune 的 `tests/test_baseline_mmlu_alignment.py` 和 `tests/test_baseline_gsm8k_alignment.py`。

运行会保存配置/源文件指纹、题池/训练抽样计划、策略、训练评测记录和汇总。AgentPrune / SafeSieve 主要查看结果目录的 `summary.json`，AgentDropout 查看 `evaluation/summary.json` 与 `evaluation/predictions.jsonl`。对比前同时检查评测完成数量和配置指纹，不把中途结果当作完整准确率。
