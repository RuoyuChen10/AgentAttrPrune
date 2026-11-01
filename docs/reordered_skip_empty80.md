# 重排序 80%：跳过完全没有入边的 Round 1

本实验保留两组已冻结的顺序和 80% 静态剪枝策略，不重新校准或归因。
只有当 Round 1 的 spatial 和 temporal 都没有保留边时，才跳过 Round 1
的全部 worker 调用，直接把 Round 0 的五个输出交给 reviewer。
只清空 spatial 或只清空 temporal 均不触发跳过。Round 0 始终执行。

两组现有 80% 策略都满足条件：各有 9 条 Round 0 spatial 边，Round 1
两种边均为 0。每题执行 5 个 worker 加 1 个 reviewer，共 6 次调用。
reviewer 输入仍按 canonical agent 身份排列，沿用匿名 ID。
trace 保留原两轮策略和顺序，另记实际执行轮次、跳过轮次和 reviewer 来源轮次。

新执行器、评估入口和启动脚本均为新增文件；原有执行器及 60%/50%
流水线不变。旧结果保留在 `evaluation/global_prune_80pct_val`，新结果写入
同一实验目录下的 `evaluation/global_prune_80pct_val_skip_empty_round1`。
配置指纹、日志、锁和断点独立。新评估仍为 Qwen3-8B、temperature=0、
2048 tokens、关闭 thinking、batch=4、同一批 1,531 道 MMLU val。
保留的边固定执行，不使用 sigmoid 或 Bernoulli。

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
# 只准备新检查点，CPU 执行。
./run_mmlu_reordered_skip_empty80.sh both prepare
# 明确选择空闲卡。两组独立进程依次执行，默认断点续跑。
CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_reordered_skip_empty80.sh both
```

`DRY_RUN=1` 只打印命令。也可把 `both` 替换为 `direct` 或 `recalibrate`，
单独继续其中一组。日志位于 `logs/mmlu_reordered_skip_empty80/<branch>/`。
此脚本不停止或重启其他 GPU 上的任务，不修改旧版汇总文件。
