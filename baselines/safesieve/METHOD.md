# SafeSieve 最小迁移版

本目录来自本机 `SafeSieve` 工作树，包含已有的 AgentAttrPrune 对齐实现。上游地址与每个原文件的 SHA256 见 `SOURCE_MANIFEST.json`；运行不需要原 `SafeSieve` 目录。共享的 prompt、HFChat、数据/参考轨迹校验来自 `BASELINES_ATTRPRUNE_ROOT` 指定的 AgentAttrPrune（默认使用当前仓库根目录）。

## 保留的方法

- `SafeSieve/graph/graph.py`、`node.py`、`pruning_algorithm.py` 保留原文件：空间边 Bernoulli 采样、时间边环检测、ZeroExtensionPruner、动态阈值和孤立节点移除。
- `edge_matrix.py` 保留角色相似度与成功回答边的互补性统计、加权打分和归一化；仅删除未使用的相似度计算导入。
- 两个 aligned runner 保留已有统计更新、剪枝日程、断点和生成审计。GSM8K 的 `score_answer`、`remove_isolated_nodes`、`parse_similarity_score` 原样提取到 `experiments/run_gsm8k.py`。
- 使用当前 AgentAttrPrune 的 MMLU/GSM8K worker/reviewer prompt；MMLU 五个 worker、GSM8K 四个 worker，两轮执行，temperature=0、max_new_tokens=2048、enable_thinking=False。LLM 微批次和评测 batch 均为 4。
- MMLU：原 seed0 dev40 循环产生 285 次训练抽样，逻辑训练 batch=16，共 18 批；第 10、15 批可触发 25% 空间剪枝。
- GSM8K：原 seed0 train40 各执行一次，逻辑训练 batch=4，共 10 批；第 10 批可触发 25% 空间剪枝。
- 原算法训练不执行 optimizer.step。保留原解析器和训练时的无效 MMLU 答案随机回退，未替换为 AgentAttrPrune 的评分规则。

## 必需的小型资源

`assets/mmlu_similarity.log` 是历史日志中五个角色、相似度模型和 5×5 矩阵的原样摘录。历史日志只打印两位小数，不能声称与当时内存中的全精度矩阵逐位一致；这与原 aligned runner 的限制相同。

`assets/gsm8k_original/` 保存原 GSM8K protocol 配置和旧角色矩阵，仅用于核对原实验预算/角色来源。其中历史绝对路径是来源记录，运行时不会读取。真正使用的是 `assets/gsm8k_aligned_similarity.json`，内含对齐后的角色描述、六组 expert 输入/输出和完整矩阵。每次加载均校验 prompt、分数和矩阵对应关系。迁移版不包含远端 API 后端；缺少或更改 cache 会报错，可用 `--similarity-json` 指定经同样验证的缓存。

未复制模型、数据集、训练结果或原始完整日志；未迁移其他任务的 agents、web/code tools、远端 API client 和重复 HFChat。源目录没有 LICENSE 文件，未自行添加授权声明。依赖为 AgentAttrPrune 已有环境加 `class-registry==2.1.2`。

## 入口和验证

统一入口与环境安装见上级 README。原模块入口也保留：

```bash
cd baselines/safesieve
export PYTHONPATH="$PWD:${BASELINES_ATTRPRUNE_ROOT:-$(dirname "$(dirname "$PWD")")}"
python -m experiments.run_mmlu_attrprune_aligned --dry-run
python -m experiments.run_gsm8k_attrprune_seed0 --dry-run
python -m pytest -q tests
```

两个入口均支持 `--stage prepare|train|eval|summarize|all`、`--run-dir`、`--reference-run`、`--resume`。`--dry-run` 只校验，不加载模型和创建结果。默认输出位于本目录 `result/`；统一入口可另外指定结果目录。训练或评测需要显式设置 `CUDA_VISIBLE_DEVICES`。

离线测试覆盖逐角色 prompt、reviewer 别名和节点删减、真实 HFChat 微批队列配假模型、原训练预算、策略/统计恢复、断点续跑、trace 完整性和相似度审计。测试不加载权重、不调用 API。迁移验证不代表已重新完成整套 GPU 对比实验。
