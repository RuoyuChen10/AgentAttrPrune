# MMLU：行和减列和定序的两个对照实验

两组都读取完整的 `dev40_random_order1_correctness_aware`。每轮使用原始 spatial `token_ratio` 的 `conditional_mean`，按行和减列和降序排序，并列沿用 canonical agent 身份顺序。

- Round 0：Hist. → Math. → Psych. → Knowl. → Critic。
- Round 1：Math. → Hist. → Psych. → Knowl. → Critic。

顺序只计算一次，两组共用。每轮 spatial 只保留前向候选，各 10 条；Round 0 temporal 不存在，Round 1 有 25 条（含自身跨轮连接）。定序后共 45 条候选，再按重要性做 global 静态剪枝：80%、60%、50% 分别保留 9、18、23 条。比例分别作用于原矩阵，不累计剪枝；不做 sigmoid 或 Bernoulli 抽边。Reviewer 输入不参与剪枝。

`direct` 分支只投影旧矩阵，保留允许边原有的总和、观测次数和均值，不假装重新获得 40 次观测。`recalibrate` 分支在完全相同的 dev40 上按上述固定顺序重新 Dense 推理一次，然后使用原 correctness-aware + all 归因；正确节点沿用 early stop，错误节点沿用完整 insertion 及无 gold 验证。新归因不再次改变顺序。

## 运行

本机 `otml-seit` 两个终端或 tmux 窗口分别运行：

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_reordered_pruning.sh direct
```

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_reordered_pruning.sh recalibrate
```

每组内部按 80 → 60 → 50 依次评估全量 1,531 道 MMLU val。Qwen3-8B、temperature=0、2048 tokens、关闭 thinking；校准生成串行，归因 batch=4，验证 batch=4。各阶段/比例使用独立进程以释放模型显存。

本次在 `otml-seit` 创建的会话为 `mmlu_rowcol_direct_45` 和 `mmlu_rowcol_recalibrate_67`。查看运行输出：

```bash
tmux attach -t mmlu_rowcol_direct_45
tmux attach -t mmlu_rowcol_recalibrate_67
```

两个命令分别进入对应会话；按 `Ctrl+b` 后再按 `d` 可退出查看并让实验继续运行。

脚本始终启用续跑，中断后重跑原命令。`DRY_RUN=1` 仅打印命令，不运行 Python、不创建结果；CPU 准备使用 `./run_mmlu_reordered_pruning.sh direct prepare`（另一组将 direct 换成 recalibrate）。`--stage` 的对应脚本位置参数支持 prepare、collect、attribute、aggregate、visualize、eval、summarize；direct 没有 collect/attribute/aggregate。

## 结果

新结果分别保存到 `dev40_rowminuscol_direct_correctness_aware` 和 `dev40_rowminuscol_recalibrated_correctness_aware`，现有实验不变。日志在 `logs/mmlu_reordered/{direct,recalibrate}`，包含每阶段 UTC 起止时间、退出码和耗时。

保存排序计划、来源 SHA、独立策略、逐题 trace/答案及批量信息。归因不完整、失败、trace 或配置被修改时不能继续发布策略；验证阶段的运行异常不会当作答错。归因拒绝截断；验证沿用现有规则评分并单独记录截断次数。

可视化沿用短名字，spatial 两轴按对应轮顺序，Round 1 temporal 的行按 Round 0、列按 Round 1 顺序。数据始终按 canonical 身份存储；展示重排不改变 source/target 对应关系。
