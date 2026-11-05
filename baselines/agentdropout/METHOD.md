# AgentDropout 精简迁移

本目录来自原同级 `AgentDropout` 项目中已经对齐的 MMLU dev40 / GSM8K seed0 train40 实现。运行不读取原 AgentDropout checkout；依赖当前 AgentAttrPrune 的 prompt、冻结参考实验和数据身份校验。`SOURCE_MANIFEST.json` 保存迁移时每个源文件的 SHA-256 和 Git HEAD（包含源项目已有的未提交修改）。

## 保留的方法

- `AgentDropout/graph/graph.py` 与 `graph/node.py` 原样保留：按轮的节点选择、Bernoulli 边采样、空间图防环、FIFO 调度及时间自连接。
- `experiments/aligned_train_mmlu.py` 原样保留：先训练节点删除，再训练边权；Adam、正确率奖励的策略梯度、核范数与 Frobenius hinge 正则。
- `experiments/run_gsm8k_aligned.py` 仅保留原训练函数、答案解析及结果读写函数；删除未被两个入口使用的旧 CLI、dense 模式和构图入口。
- `experiments/*prompt_adapter.py` 直接调用 AgentAttrPrune 的 worker/reviewer 消息构建函数；保留 AgentDropout 的 dropout sentinel 处理和原答案评分。GSM8K 对齐版不会执行 Programming Expert 输出中的代码，沿用已有对齐实现。
- `AgentDropout/llm/hf_chat.py` 原样保留本方法的队列批处理 / 模型副本后端。它与其他方法共享解码参数，但推理调度不是 AgentAttrPrune HFChat。每次生成最多 4 个请求，默认 2 个模型副本；`--replicas 1` 可使用一个副本。每个副本完整驻留于一张 GPU，不支持模型跨 GPU 切分。

| 数据集 | 校准池 | 节点训练 | 边训练 | 剪枝事件 | 评测 |
| --- | --- | --- | --- | --- | --- |
| MMLU | 原 seed0 dev40 | 10 × 16 = 160 draws | 10 × 16 = 160 draws | 边更新 5、10 后各调用原 25% 剪枝规则 | val 全部 1531 题 |
| GSM8K | 原 seed0 train40 | 2 × 20 = 40 draws | 4 × 10 = 40 draws | 边更新 2、4 后各调用原 10% 剪枝规则 | test 全部 1319 题 |

两者均为 seed=0、2 rounds、temperature=0、max_new_tokens=2048、thinking=false；MMLU 5 workers、GSM8K 4 workers，另有 reviewer。默认评测 batch=4、LLM batch=4。**训练逻辑批次保留上表数值，不改为 4。** 评测使用原随机图策略，每题独立 RNG；保留原 parser 的行为，不将评分器替换成 AgentAttrPrune 的 parser。

## 入口与依赖

优先用上层统一 launcher；直接模块入口是：

```bash
# 从 AgentAttrPrune 根目录进入本目录；默认共享代码在当前仓库根目录。
cd baselines/agentdropout
export BASELINES_ATTRPRUNE_ROOT="${BASELINES_ATTRPRUNE_ROOT:-$(dirname "$(dirname "$PWD")")}"
export PYTHONPATH="$PWD:$BASELINES_ATTRPRUNE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.run_mmlu_attrprune_dev40 --stage prepare --dry-run
python -m experiments.run_gsm8k_attrprune_seed0 --stage prepare --dry-run
```

`--stage` 支持 `prepare/train/eval/summarize/all`，也保留旧的位置参数形式。`--reference-run` 指向完整原始参考实验；`--run-dir` 指向新的结果目录。模型默认从参考实验的 `model_name` 读取，也可传 `--model-path`。`--dry-run` 只读验证，不加载模型、不写结果。训练和评测需显式设置 `CUDA_VISIBLE_DEVICES`（统一 launcher 会检查）。

依赖 AgentAttrPrune 已有环境，再加 `class-registry`（源环境版本 2.1.2）；还使用 `shortuuid/numpy/torch/transformers/pandas/matplotlib`。不依赖 Hugging Face `datasets`、OpenAI API、搜索工具或代码执行器。此目录自己的 `datasets` 只含轻量解析代码，不能与其他方法的同名 `experiments` 包放进一个 Python 进程混用。

保留原有恢复边界：`--resume` 可复用完整训练 checkpoint 并补齐评测；中途训练尚无 optimizer checkpoint 时需新建 run 目录，不能直接续训。新目录迁移后的 source hash 与旧目录不同，不能直接写回原结果目录。

## 校验

`tests/test_mmlu_attrprune_dev40.py` 和 `tests/test_gsm8k_attrprune_seed0.py` 从源项目迁移，覆盖 mock LLM 下完整训练、图采样、checkpoint、恢复、生成失败传播、batch=4 与逐题评测一致性、真实 HF 队列四请求批处理，以及冻结参考的 440/360 条 prompt 一致性。运行不会加载大模型。

```bash
python -m pytest tests/test_mmlu_attrprune_dev40.py tests/test_gsm8k_attrprune_seed0.py -q
```

## 来源

代码来源项目 README 标注的上游为 [AgentDropout](https://github.com/wangzx1219/AgentDropout)，论文为 Wang et al., *AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration*, ACL 2025。框架致谢 GPTSwarm 与 AgentPrune。迁移时本地源目录没有 LICENSE 文件；本目录没有重新指定许可证。
