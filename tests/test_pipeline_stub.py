from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.attribution import AttributionConfig, EdgeAttributionCalibrator
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    name_or_path = "stub-character-tokenizer"

    def __len__(self):
        return 256

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        del tokenize, kwargs
        rendered = "".join(
            f"<{item['role']}>{item['content']}</{item['role']}>"
            for item in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(
        self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs
    ):
        del add_special_tokens, kwargs
        values = [ord(char) % 251 + 1 for char in text]
        payload = {"input_ids": values}
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return payload

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr((int(value) - 1) % 251) for value in token_ids)


class TraceStubLLM:
    model_name = "stub"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def agen(
        self, messages, *, max_tokens=256, temperature=0.0, num_comps=1
    ):
        is_reviewer = "top decision-maker" in messages[0]["content"]
        output = (
            "A"
            if is_reviewer
            else "First useful evidence sentence. Second useful evidence sentence."
        )
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(rendered)["input_ids"]
        generated_ids = self.tokenizer(output)["input_ids"]
        observer = generation_observer.get()
        if observer is not None:
            observer(
                {
                    "model_name": self.model_name,
                    "messages": messages,
                    "rendered_prompt": rendered,
                    "input_ids": [input_ids],
                    "attention_mask": [[1] * len(input_ids)],
                    "generated_ids": [generated_ids],
                    "output_text": [output],
                    "generation_kwargs": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "num_completions": num_comps,
                    },
                }
            )
        return output


class VisibleTokenScorer:
    model_name = "stub"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = SimpleNamespace(config=SimpleNamespace(_name_or_path="stub"))

    def score_teacher_forced_targets(
        self,
        prompt_input_ids,
        generated_ids=None,
        selected_output_token_indices=None,
        attention_mask=None,
        score_type="logit",
        reduction="sum",
        **kwargs,
    ):
        del generated_ids, selected_output_token_indices, score_type, reduction, kwargs
        rows = prompt_input_ids
        masks = attention_mask
        scores = []
        for row, mask in zip(rows, masks):
            row_tensor = torch.as_tensor(row, dtype=torch.float32)
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
            scores.append((row_tensor * mask_tensor).sum())
        score = torch.stack(scores)
        return {
            "score": score,
            "logit_sum": score,
            "logit_mean": score,
            "logprob_sum": score - 1,
            "logprob_mean": score - 1,
        }


def test_schema_v1_trace_to_attribution_policy_to_graph_replay(tmp_path):
    tokenizer = CharacterTokenizer()
    specs = [WorkerSpec("a", "Critic"), WorkerSpec("b", "Mathematician")]
    trace_path = tmp_path / "calibration" / "traces" / "sample_00000" / "trace.json"
    recorder = GenerationTraceRecorder(
        trace_path,
        config={
            "model_name": "stub",
            "fingerprint": "stub-run-fingerprint",
            "sample_index": 0,
            "num_rounds": 1,
            "workers": [
                {
                    "agent_id": spec.id,
                    "role": spec.role,
                    "agent_type": spec.agent_type,
                    "node_index": index,
                }
                for index, spec in enumerate(specs)
            ],
        },
        sample={"task": "Choose A.", "answer": "A"},
    )
    dense_graph = MultiAgentGraph(TraceStubLLM(tokenizer), specs, num_rounds=1)
    dense_result = asyncio.run(
        dense_graph.run_sample(
            {"task": "Choose A.", "answer": "A"},
            recorder=recorder,
            phase="calibration",
            sample_index=0,
        )
    )
    final = dense_result.to_trace_final()
    final.update({"parsed_answer": "A", "gold_answer": "A", "correct": True})
    recorder.write(final)

    output_dir = tmp_path / "attribution"
    calibrator = EdgeAttributionCalibrator(
        run_dir=trace_path.parents[1],
        output_dir=output_dir,
        model_name="stub",
        scorer=VisibleTokenScorer(tokenizer),
        config=AttributionConfig(
            min_incoming=1,
            min_selected_sentences=1,
            min_sentence_words=1,
            target_recovery=0.8,
            policy_pruning_rate=0.0,
        ),
    )
    calibrated = calibrator.run()

    assert calibrated["aggregate"]["num_correct_samples"] == 1
    sample_dir = output_dir / "sample_00000"
    assert (sample_dir / "sample_edge_matrices.json").is_file()
    assert (sample_dir / "sample_token_ratio_matrices.png").is_file()
    index = json.loads(
        (sample_dir / "node_attribution_index.json").read_text(encoding="utf-8")
    )
    assert index["expected_event_count"] == 3
    assert index["actual_event_count"] == 3
    assert index["complete"] is True
    assert [item["status"] for item in index["events"]] == [
        "no_attributable_input",
        "attributed",
        "attributed",
    ]
    for item in index["events"]:
        event = json.loads(Path(item["json"]).read_text(encoding="utf-8"))
        assert "messages" in event["input"]
        assert "text" in event["output"]
        assert item["visualization"] == item["svg"]
        svg_path = Path(item["svg"])
        assert svg_path.is_file()
        svg = svg_path.read_text(encoding="utf-8")
        assert "Selected Incoming-Sentence Attribution" in svg
        assert "Sentence-level" in svg
        assert 'id="saliency-gradient"' in svg
        assert "1.0 top1" in svg
        assert "Token-level" not in svg
        curve_path = Path(item["insertion_curve_svg"])
        assert curve_path.is_file()
        curve_svg = curve_path.read_text(encoding="utf-8")
        assert "Early-stop Insertion Score Curve" in curve_svg
        if item["status"] == "no_attributable_input":
            assert "no insertion search was required" in curve_svg
        else:
            assert "Retained Significant Sentences (%)" in curve_svg
            assert "Insertion Score (logit · sum)" in curve_svg
            assert "stop:" in curve_svg
        assert Path(item["html"]).is_file()
        saliency_configuration = event["attribution"]["saliency_configuration"]
        assert saliency_configuration["floor"] == 0.4
        selected = sorted(
            (
                sentence
                for sentence in event["sentences"]
                if sentence["selected"]
            ),
            key=lambda sentence: sentence["selection_step"],
        )
        unselected = [
            sentence
            for sentence in event["sentences"]
            if not sentence["selected"]
        ]
        if selected:
            assert selected[0]["saliency"] == 1.0
            expected_last = 1.0 if len(selected) == 1 else 0.4
            assert selected[-1]["saliency"] == expected_last
        assert all(sentence["saliency"] is None for sentence in unselected)

    first_svg = Path(index["events"][0]["svg"])
    first_svg.unlink()
    rendered = calibrator.run(render_only=True)
    assert len(rendered["manifest"]["samples"]) == 1
    assert first_svg.is_file()
    assert (output_dir / "visualization_manifest.json").is_file()
    assert (output_dir / "pruned_policy_binary.json").is_file()
    assert (output_dir / "pruned_policy_token_ratio.json").is_file()
    assert (output_dir / "pruned_policy_word_ratio.json").is_file()

    policy = RoundPolicy.from_json(output_dir / "pruned_policy_binary.json")
    replay = MultiAgentGraph(
        TraceStubLLM(tokenizer), specs, num_rounds=1, policy=policy
    )
    replay_result = asyncio.run(replay.run_sample({"task": "Choose A."}))
    assert replay_result.final_answer == "A"
