# MMLU dev40：每题一次随机顺序

这个入口把预算缩减为同一批 40 道题、每题一次完整两轮推理，共 40 次。每题的 Round 0 和 Round 1 仍分别独立随机排列 5 个 worker；spatial、temporal、correctness-aware 归因和按实际出现次数求均值的定义不变。不生成剪枝策略，不运行 val。

直接选取已冻结的 400 次计划中每题的 `repeat_00`，不按答案正确性或归因效果选择 repeat。全部 40 个原始生成 trace 已存在，因此不需要重新生成。原 400 次目录和固定顺序参考目录均保留。

新目录是 `results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware`。新目录中的预算和配置指纹属于 40 次实验；复制时不改变节点的输入、输出、排列和 token ID。来源 SHA 与缓存迁移信息另外记录。只有与原 trace、算法和归因配置一致的完整节点缓存可以复用；后续归因补齐缺失节点，然后重新校验和聚合。

## 运行

在 `otml-seit2` 的原 tmux pane 中先用 **Ctrl-C** 停止 400 次归因，等待 shell 提示符出现，再执行：

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_dev40_random_order_once.sh
```

默认 `calibrate` 依次进行 CPU 数据准备、补齐归因、聚合和可视化。每题只保留一次、归因 batch=4、关闭 thinking、temperature=0、2048 tokens。已存在的合法目标缓存会保留，重复执行同一命令即可续跑。旧 400 次进程未退出时，GPU 阶段会拒绝启动，避免同时占用设备。

只准备数据或检查命令时：

```bash
./run_mmlu_dev40_random_order_once.sh prepare
DRY_RUN=1 ./run_mmlu_dev40_random_order_once.sh
```

也可以分别运行 `attribute`、`aggregate`、`visualize`。只有 `attribute` 和 `calibrate` 需要指定 GPU。`prepare` 可在旧任务仍在运行时执行，只读取已经保存的源结果。

## 日志与结果

每次启动创建独立日志 `logs/mmlu_dev40_random_order1/<stage>_<UTC时间>_<PID>.log`，开头打印准确路径。日志记录终端输出、各阶段开始和结束时间、耗时秒数及退出状态。时间标记使用 UTC，北京时间加 8 小时。

新目录下 `attribution/manifest.json` 记录归因完成状态；只有 40 次全部归因成功、`aggregation_complete=true` 且 aggregate SHA 匹配时才发布完整矩阵。主矩阵仍是行 source、列 target，缺失边不参与均值；出现但纠错失败的边以零值参与均值。

验收时，每轮任意两个 agent 的两个 spatial 方向出现次数之和必须为 **40**，Round 1 temporal 的每格出现次数必须为 **40**。Round 0 temporal 不存在。各方向重要性可以为零，不能用正值要求代替完整性校验。

原 `run_mmlu_dev40_random_order.sh` 默认仍保留 40×10 的实验。底层 Python runner 增量支持 `--repeats 1`，但不允许在已存在的 400 次目录中改变预算。
