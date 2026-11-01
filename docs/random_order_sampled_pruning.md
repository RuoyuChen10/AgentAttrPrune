# 逐轮归因矩阵：剪枝后按 AgentPrune 流程采样

来源是完整的 `mmlu/qwen3_8b/dev40_random_order1_correctness_aware`：同一批 40 道校准题、每题一组两轮随机顺序。评估采用 MMLU val，默认全量 1,531 题，每题一次完整两轮推理。

保留 Round 0 spatial、Round 1 spatial、Round 1 temporal 三张独立归因矩阵。每个已观测候选的 `token_ratio / conditional_mean` 直接作为 logit，采样概率为 `sigmoid(importance)`，采样温度为 1。这里没有训练 AgentPrune 的 logits，也没有把重要性直接当作概率。

先按重要性一次性裁掉候选，然后每题每轮重新采样实际图：

- **global**：对 20 + 20 + 25 = 65 个轮次候选统一排序剪枝。
- **uniform**：三张矩阵分别排序，各剪同一比例；不表示每个 agent 都保留相同度数。
- 50% 与 60% 分别从原始矩阵开始，不连续累计剪枝。沿用 `torch.round` 的取整约定，50% 时实际移除 32/65 条候选，60% 时移除 39/65。
- Spatial 按固定身份的 source-major 顺序枚举；先检查是否成环，再按 Bernoulli 抽边。成环、被硬剪掉的边不消耗采样随机数。执行顺序由 FIFO 拓扑排序得到，不额外随机打乱 worker。
- Round 0 没有 temporal。Round 1 temporal 独立采样，允许跨轮自身连接，不做 spatial 拒环检查。
- Reviewer 读取末轮全部 worker，固定身份和匿名 ID 沿用校准实验；它的输入边不参加剪枝或采样。

这些采样和执行规则与本地 AgentPrune 一致。用户选择保留逐轮矩阵，因此没有采用 AgentPrune 原版跨轮共享参数的结构；global 分配也是本实验的变体。输出记录完整定义，不能把结果称为重新训练的原版 AgentPrune。

## 在 otml-seit2 运行

在两个 tmux 窗口分别运行；每对卡内 50%、60% 顺序执行：

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_random_order_sampling.sh global 50 60
```

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_random_order_sampling.sh uniform 50 60
```

不再跑校准生成或归因，只读取其冻结结果并评估。运行前应确保对应设备上的旧任务已经退出。默认 Qwen3-8B、两轮、5 个 worker、temperature=0、2048 tokens、关闭 thinking；一个进程使用所指定的两张卡，问题串行执行。

```bash
# CPU 检查命令，不写文件、不加载模型
DRY_RUN=1 ./run_mmlu_random_order_sampling.sh global 50 60

# CPU 准备冻结策略和评估清单
PREPARE_ONLY=1 ./run_mmlu_random_order_sampling.sh uniform 50 60
```

默认续跑；重复命令只处理尚未完成的题。`VAL_LIMIT` 可改变题量，`SEED` 控制图采样；若改变已冻结的实验配置，需用 `OUTPUT_ROOT` 指定独立目录。每题的 seed 不包含剪枝比例或分配方式，但被跳过的候选不消耗 RNG，所以不同策略不保证每条公共边获得同一个随机数。

跨服务器续跑会原样复用已冻结策略。仅供记录的批量 sigmoid 概率允许不同 CPU 指令集造成的一个 float32 ULP 舍入差异；策略 SHA、logits、剪枝 mask、数据和其他配置仍需通过校验，不重写已有策略或指纹。实际采样使用标量 sigmoid。

## 日志与结果

日志位于 `logs/mmlu_random_order_sampling/<global|uniform>/`，每次执行、每个剪枝率单独保存，包含 UTC 起止时间和退出码。终端会打印准确的输出目录。

结果位于源实验的独立 `evaluation/agentprune_sampling/per_round_sigmoid/` 子目录，保存来源 SHA、候选 mask、logits/概率、逐题 seed、实际采样 mask 和执行顺序、trace、答案及完整性状态。汇总同时报告候选剪枝率和实际消息边数；Bernoulli 与拒环会进一步减少边数，实际执行图的稀疏度并不恒等于 50% 或 60%。
