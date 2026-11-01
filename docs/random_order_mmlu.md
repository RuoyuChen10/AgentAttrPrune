# MMLU dev40 随机顺序 Dense 归因

这是独立的增量实验：沿用已有 dev40 correctness-aware 实验的 40 道题、模型、提示词、角色、两轮和生成设置，每题重新执行 10 次，共 400 次完整 Dense 推理。目标是先观察解除固定 worker 顺序后的边重要性；本入口只收集、归因、聚合及可视化，不生成剪枝策略，也不运行 val 评估。本文描述实验协议和运行方法，不表示 400 次正式实验已经运行完成。

## 顺序与通信

- 每个 `(question, repeat, round)` 独立抽取一次均匀随机 worker 排列。Round 0 与 Round 1 各自抽取，允许偶然相同；不是从不重复排列中选取。`ORDER_SEED=0` 用于可复现的顺序计划。
- 在每轮排列下执行 Dense：每个 worker 接收该轮所有先于它执行的 worker 输出。没有额外 Bernoulli 采边。矩阵坐标始终按固定 agent 身份排列，不能按本次执行位置排列。
- Spatial 没有自己发给自己的边，对角线为 0；单次轨迹在执行顺序坐标下仍然无环，但固定身份坐标下累计可覆盖上下两个三角区域。
- Temporal 保持现有 Dense 语义：Round 0 没有 temporal 边，Round 1 接收上一轮所有 worker 的输出，**保留 self-temporal**。它表示“上一轮的自己发给本轮的自己”，不等于同轮自环。
- Reviewer 继续接收末轮全部 worker。匿名来源 ID 与角色身份的对应关系沿用 reference，不随排列重新分配；reviewer 边不混入 worker 矩阵。

40 × 10 的预算来自本地 **GSM8K AgentPrune** 已保存的实跑：40 个不同训练题，每题 10 次，2 轮。本地 MMLU AgentPrune 默认是 batch 4 × 10 iterations，即 40 次推理；其现有 launcher 使用 batch 16 × 10，即 160 次，并不是每题 10 次。AgentPrune 通过采边、防环和拓扑执行产生变化，本实验则直接均匀随机执行顺序；两者不能视为相同的随机图分布。本实验保留本仓 Dense 的 temperature=0，以观察顺序变化的影响。

## 归因与平均

沿用 correctness-aware 归因，并包含所有 400 次推理，不按 reviewer 最终正确性过滤。正确节点使用原有完整输出归因；错误节点使用 gold-prefix greedy insertion，取最早最高分子集，再用原始提示词、不注入 gold，验证是否能纠错。局部验证不覆盖原始输出，也不改变原轨迹中下游节点的输入。

对每个 `(round, edge_type, source, target)`，只把该边实际存在的推理纳入分母：

```text
conditional_mean = sum(importance where edge is present) / present_count
```

例如在 400 次中出现 300 次，重要性之和就除以 300。边缺失时不贡献分子或分母；边存在但 `rescue_failed` 时，重要性为 **已观测的 0**，仍计入分母。OOM、输出截断或归因异常属于运行失败，不能冒充零重要性或直接丢掉后发布完整矩阵。

可视化需要结合重要性与出现次数阅读。Spatial 非对角位置有观测后，即可获得估计值；有观测的估计值仍可能为 0。完全没有观测的边应显示为缺失，而不是人为填出正值。Round 0 temporal 全部结构性不存在；Round 1 temporal 包括对角线。

## 运行

在仓库根目录执行。先查看命令，不使用 GPU、不加载模型，也不写输出：

```bash
DRY_RUN=1 ./run_mmlu_dev40_random_order.sh
```

默认 reference 为 `results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware`。默认新输出为 `results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order10_correctness_aware`；reference 内容只读。输出位置可以通过 `RESULTS_ROOT`、`RUN_NAME` 更改，reference 需要单独用 `REFERENCE_RUN` 更改。

GPU 稍后由用户指定；设置实际可用设备后，才能运行完整流程或 GPU 阶段：

```bash
# 先将 CUDA_VISIBLE_DEVICES 设置为用户选定的设备，再执行：
./run_mmlu_dev40_random_order.sh calibrate

# 也可拆分运行，便于续跑和检查：
./run_mmlu_dev40_random_order.sh collect
./run_mmlu_dev40_random_order.sh attribute
./run_mmlu_dev40_random_order.sh aggregate
./run_mmlu_dev40_random_order.sh visualize
```

`calibrate`、`collect`、`attribute` 要求显式非空 `CUDA_VISIBLE_DEVICES`；`aggregate`、`visualize` 不要求 GPU。默认 `RESUME=1`，续跑时须保持 reference、输出位置、顺序 seed 等配置一致。默认 `ATTR_BATCH_SIZE=4`。40 题、每题 10 次及继承的模型/生成设置由专用 Python 入口固定，旧脚本的 `DEV_LIMIT`、`TEMPERATURE` 等环境变量不是本实验的配置入口。

`calibrate` 按阶段启动独立子进程，使生成模型在归因模型加载前释放。生成串行运行；归因 batch 指同一接收节点的候选输入评分批次，不是同时执行多个 worker。单个运行目录使用进程锁，避免两个进程同时改写 checkpoint。

默认通过 Conda 环境 `agentprune` 运行；使用指定解释器时：

```bash
CONDA_ENV='' \
PYTHON_BIN=/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python \
DRY_RUN=1 ./run_mmlu_dev40_random_order.sh
```

对应 Python 接口：

```bash
python -m experiments_attrprune.random_order_mmlu \
  --stage calibrate \
  --reference-run results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware \
  --run-dir results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order10_correctness_aware \
  --order-seed 0 --attr-batch-size 4 --resume
```

验收时应核对 40 个 question、每题 10 条 rollout、共 400 条；各 rollout 两轮排列、原始 trace、逐节点归因和矩阵均完整；缺失边和验证失败零值的计数不同；上下三角的重要性热图和对应出现次数可共同解释。只有完整归因通过校验后，才能将聚合矩阵作为正式结果使用。

## 产物与检查

以下路径均相对于新的运行目录：

| 产物 | 内容 |
|---|---|
| `run_config.json`、`rollout_plan.json` | 参考来源指纹、统一实验配置、400 个复合身份及两轮排列 |
| `calibration/collection_state.json` | 已完成 rollout 的 SHA、失败诊断及续跑进度 |
| `calibration/selection_manifest.json` | 所有 400 条完整 trace 的冻结清单；全部生成完成后发布 |
| `calibration/traces/sample_00000_repeat_00/trace.json` | 一次推理的完整原始轨迹；题号和 repeat 独立保存 |
| `attribution/sample_00000_repeat_00/` | 逐节点 JSON/HTML/SVG、纠错诊断、样本矩阵和完成索引 |
| `attribution/aggregate_edge_matrices.json` | 按 round/source/target 保存的 sum、opportunity_count、conditional_mean，及覆盖率/节点状态统计 |
| `attribution/visualizations/random_order_conditional_mean.*` | 两轮 spatial/temporal 重要性热图，PNG 和 SVG |
| `attribution/visualizations/random_order_opportunity_count.*` | 每条边实际出现次数热图 |
| `attribution/visualizations/random_order_pooled_spatial.*`、`pooled_spatial.json` | 跨轮 spatial 总重要性除以总出现次数 |
| `attribution/visualizations/random_order_vs_fixed_order.*` | 随机顺序400次与固定顺序40次对照，共用0–1色阶 |
| `attribution/visualizations/random_order_edges.csv` | 逐边总和、次数、均值、是否观测和是否结构允许 |

正式聚合必须满足 `attribution/manifest.json` 的 `complete=true`、`aggregation_complete=true`、无 failures，且 `aggregate_sha256` 匹配矩阵文件。单看文件存在不能判断成功。每轮 spatial 任意两个不同 agent 的正反方向出现次数之和应为 400；Round 1 temporal 每格应为 400。非对角边出现次数不足时如实报告，不额外补采、填零或改变原有采样预算。

运行 CPU 回归测试：

```bash
CUDA_VISIBLE_DEVICES='' \
  /egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python -m pytest -q
```

测试使用 stub 模型验证顺序、来源、重复推理身份、断点恢复、失败零值及出图；不替代真实 Qwen3-8B 的400次实验。正式实验运行后，重要性图才能作为方法结果解释。
