import inspect

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TeacherForcedTextScorer(nn.Module):
    def __init__(self, model, tokenizer, model_name="Qwen/Qwen3-8B", device=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device or getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.softmax = nn.Softmax(dim=-1)

        self.prompt_length = None
        self.generated_answer_ids = None
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is not None:
            self.mask_token_id = self.tokenizer.pad_token_id
        else:
            self.mask_token_id = self.tokenizer.eos_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer needs either pad_token_id or eos_token_id for text masking.")

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, model_name=model_name)

    def tokenize_rendered_chat(self, rendered_text):
        return self.tokenizer(
            [rendered_text],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

    def generate_from_rendered_chat(self, rendered_text, max_new_tokens=128, do_sample=False):
        inputs = self.tokenize_rendered_chat(rendered_text)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=do_sample,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )

        prompt_length = inputs["input_ids"].shape[1]
        answer_ids = generated_ids[0, prompt_length:]
        output_text = self.tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_tokens = [
            self.tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for token_id in answer_ids
        ]

        self.prompt_length = prompt_length
        self.generated_answer_ids = answer_ids.detach().clone()
        self.target_token_position = torch.arange(
            prompt_length,
            prompt_length + len(answer_ids),
            device=self.model.device,
            dtype=torch.long,
        )
        self.selected_interpretation_token_word_id = answer_ids.detach().clone()

        return {
            "input_ids": inputs["input_ids"][0].detach().cpu(),
            "attention_mask": inputs["attention_mask"][0].detach().cpu(),
            "generated_ids": generated_ids.detach().cpu(),
            "generated_answer_ids": answer_ids.detach().cpu(),
            "output_text": output_text,
            "output_tokens": output_tokens,
        }

    def set_targets_from_rendered_chat_output(
        self,
        rendered_text,
        output_text,
        selected_output_token_indices=None,
    ):
        inputs = self.tokenize_rendered_chat(rendered_text)
        answer_ids = self.tokenizer(
            output_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(self.model.device)
        if answer_ids.numel() == 0:
            raise ValueError("output_text produced no target tokens.")

        self.prompt_length = inputs["input_ids"].shape[1]
        self.generated_answer_ids = answer_ids.detach().clone()
        self.target_token_position = torch.arange(
            self.prompt_length,
            self.prompt_length + len(answer_ids),
            device=self.model.device,
            dtype=torch.long,
        )
        self.selected_interpretation_token_word_id = answer_ids.detach().clone()
        self.set_targets(answer_ids, selected_output_token_indices)

        output_tokens = [
            self.tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for token_id in answer_ids
        ]
        return {
            "input_ids": inputs["input_ids"][0].detach().cpu(),
            "attention_mask": inputs["attention_mask"][0].detach().cpu(),
            "generated_answer_ids": answer_ids.detach().cpu(),
            "output_text": output_text,
            "output_tokens": output_tokens,
        }

    def set_targets(self, generated_answer_ids, selected_output_token_indices=None):
        if self.prompt_length is None:
            raise ValueError("prompt_length is not set. Run generate_from_rendered_chat first.")

        answer_ids = torch.as_tensor(generated_answer_ids, device=self.model.device, dtype=torch.long)
        if selected_output_token_indices is None:
            selected_output_token_indices = list(range(len(answer_ids)))
        else:
            selected_output_token_indices = [
                int(index)
                for index in selected_output_token_indices
                if 0 <= int(index) < len(answer_ids)
            ]
            if not selected_output_token_indices:
                raise ValueError("No selected output token indices are within the generated answer.")
        selected_indices = torch.as_tensor(
            selected_output_token_indices,
            device=self.model.device,
            dtype=torch.long,
        )
        self.generated_answer_ids = answer_ids
        self.target_token_position = self.prompt_length + selected_indices
        self.selected_interpretation_token_word_id = answer_ids[selected_indices]

    @staticmethod
    def _as_sequence_rows(value, name):
        """Normalize a single sequence or a (possibly ragged) batch to 1-D tensors."""
        if isinstance(value, torch.Tensor):
            if value.dim() == 1:
                return [value]
            if value.dim() == 2:
                return [row for row in value]
            raise ValueError(f"{name} must be one- or two-dimensional.")

        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"{name} is empty.")
            first = value[0]
            first_is_sequence = isinstance(first, (list, tuple, torch.Tensor)) or (
                hasattr(first, "ndim") and int(first.ndim) > 0
            )
            if first_is_sequence:
                rows = [torch.as_tensor(row) for row in value]
                if any(row.dim() != 1 for row in rows):
                    raise ValueError(f"Every row in {name} must be one-dimensional.")
                return rows
            return [torch.as_tensor(value)]

        tensor = torch.as_tensor(value)
        if tensor.dim() == 1:
            return [tensor]
        if tensor.dim() == 2:
            return [row for row in tensor]
        raise ValueError(f"{name} must be one- or two-dimensional.")

    @staticmethod
    def _supports_forward_argument(model, argument):
        try:
            return argument in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _prepare_teacher_forced_batch(self, prompt_input_ids, attention_mask, generated_ids, max_prefix_len):
        prompt_rows = self._as_sequence_rows(prompt_input_ids, "prompt_input_ids")
        if attention_mask is None:
            mask_rows = [torch.ones_like(row, dtype=torch.long) for row in prompt_rows]
        else:
            mask_rows = self._as_sequence_rows(attention_mask, "attention_mask")
            if len(mask_rows) != len(prompt_rows):
                raise ValueError("attention_mask batch size does not match prompt_input_ids.")

        compact_prompts = []
        for row, mask in zip(prompt_rows, mask_rows):
            row = row.reshape(-1)
            mask = mask.reshape(-1).to(row.device)
            if row.numel() != mask.numel():
                raise ValueError("Each attention_mask row must match its prompt_input_ids row.")
            compact = row[mask.bool()].to(dtype=torch.long)
            if compact.numel() == 0:
                raise ValueError("Every prompt must contain at least one visible token.")
            compact_prompts.append(compact)

        device = self.model.device
        prompt_lengths = torch.as_tensor(
            [row.numel() for row in compact_prompts], device=device, dtype=torch.long
        )
        max_prompt_len = int(prompt_lengths.max().item())
        batch_size = len(compact_prompts)
        total_length = max_prompt_len + int(max_prefix_len)
        input_ids = torch.full(
            (batch_size, total_length),
            int(self.mask_token_id),
            device=device,
            dtype=torch.long,
        )
        batch_attention_mask = torch.zeros_like(input_ids, dtype=torch.long)

        for row_index, prompt in enumerate(compact_prompts):
            prompt = prompt.to(device=device, dtype=torch.long)
            start = max_prompt_len - prompt.numel()
            input_ids[row_index, start:max_prompt_len] = prompt
            batch_attention_mask[row_index, start:max_prompt_len] = 1

        if max_prefix_len:
            prefix = generated_ids[:max_prefix_len].to(device=device, dtype=torch.long)
            input_ids[:, max_prompt_len:] = prefix.unsqueeze(0)
            batch_attention_mask[:, max_prompt_len:] = 1

        position_ids = batch_attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(batch_attention_mask == 0, 0)
        return input_ids, batch_attention_mask, position_ids, prompt_lengths, max_prompt_len

    def score_teacher_forced_targets(
        self,
        prompt_input_ids,
        generated_ids=None,
        selected_output_token_indices=None,
        attention_mask=None,
        score_type="logit",
        reduction="sum",
        use_logits_to_keep=True,
    ):
        """Score fixed generated targets under one or more prompt variants.

        ``prompt_input_ids`` may be a single 1-D sequence, a rectangular tensor,
        or a ragged list of sequences. Zero entries in ``attention_mask`` are
        treated as deleted prompt positions and compacted before left-padding the
        batch. A single ``generated_ids`` sequence is teacher-forced for every
        prompt variant. The method does not mutate targets previously installed
        with :meth:`set_targets`.

        The returned tensors always retain a batch dimension. ``target_logits``,
        ``target_logprobs``, and ``target_probabilities`` are float32 tensors of
        shape ``[batch, selected_targets]``. ``score`` follows ``score_type`` and
        ``reduction``; all sum/mean variants are also returned explicitly.
        """
        if generated_ids is None:
            generated_ids = self.generated_answer_ids
        if generated_ids is None:
            raise ValueError("generated_ids is required when generation targets are not initialized.")

        generated_ids = torch.as_tensor(generated_ids, device=self.model.device, dtype=torch.long)
        if generated_ids.dim() == 2 and generated_ids.shape[0] == 1:
            generated_ids = generated_ids.squeeze(0)
        if generated_ids.dim() != 1 or generated_ids.numel() == 0:
            raise ValueError("generated_ids must be a non-empty one-dimensional sequence.")

        if selected_output_token_indices is None:
            selected_output_token_indices = list(range(generated_ids.numel()))
        selected_output_token_indices = [int(index) for index in selected_output_token_indices]
        if not selected_output_token_indices:
            raise ValueError("At least one output token must be selected for scoring.")
        if any(index < 0 or index >= generated_ids.numel() for index in selected_output_token_indices):
            raise ValueError("selected_output_token_indices contains an out-of-range index.")

        if score_type not in {"logit", "logprob", "probability"}:
            raise ValueError(f"Unsupported score_type: {score_type}")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        selected_indices = torch.as_tensor(
            selected_output_token_indices,
            device=self.model.device,
            dtype=torch.long,
        )
        max_prefix_len = int(selected_indices.max().item())
        (
            input_ids,
            batch_attention_mask,
            position_ids,
            prompt_lengths,
            max_prompt_len,
        ) = self._prepare_teacher_forced_batch(
            prompt_input_ids,
            attention_mask,
            generated_ids,
            max_prefix_len,
        )

        logit_positions = max_prompt_len + selected_indices - 1
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": batch_attention_mask,
            "return_dict": True,
            "use_cache": False,
        }
        if self._supports_forward_argument(self.model, "position_ids"):
            model_kwargs["position_ids"] = position_ids

        supports_logits_to_keep = bool(
            use_logits_to_keep
            and self._supports_forward_argument(self.model, "logits_to_keep")
        )
        trailing_positions = torch.arange(
            input_ids.shape[1] - len(logit_positions),
            input_ids.shape[1],
            device=logit_positions.device,
        )
        positions_are_trailing = bool(torch.equal(logit_positions, trailing_positions))
        device_map = getattr(self.model, "hf_device_map", None)
        is_sharded = isinstance(device_map, dict) and len(set(device_map.values())) > 1
        used_logits_to_keep = supports_logits_to_keep and (
            positions_are_trailing or not is_sharded
        )
        if used_logits_to_keep:
            if positions_are_trailing:
                # An integer means "keep the last N logits" and cannot be
                # moved to the wrong GPU by Accelerate's input hook.
                model_kwargs["logits_to_keep"] = len(logit_positions)
            else:
                # Arbitrary index tensors are safe for unsharded models. For a
                # sharded model we fall back to full logits above because
                # Accelerate moves this kwarg to the first GPU while Qwen3
                # indexes final hidden states on the lm_head GPU.
                model_kwargs["logits_to_keep"] = logit_positions

        with torch.no_grad():
            outputs = self.model(**model_kwargs)

        logits = outputs.logits
        if used_logits_to_keep:
            if logits.shape[1] != len(selected_output_token_indices):
                raise ValueError("Model returned an unexpected sequence length for logits_to_keep.")
            selected_position_logits = logits
        else:
            selected_position_logits = logits[:, logit_positions]

        selected_position_logits = selected_position_logits.to(torch.float32)
        target_token_ids = generated_ids[selected_indices]
        gather_indices = target_token_ids.to(selected_position_logits.device).unsqueeze(0).unsqueeze(-1).expand(
            input_ids.shape[0], -1, -1
        )
        target_logits = selected_position_logits.gather(dim=2, index=gather_indices).squeeze(-1)
        target_logprobs = torch.log_softmax(selected_position_logits, dim=-1).gather(
            dim=2, index=gather_indices
        ).squeeze(-1)
        target_probabilities = target_logprobs.exp()

        metric_values = {
            "logit": target_logits,
            "logprob": target_logprobs,
            "probability": target_probabilities,
        }
        selected_metric = metric_values[score_type]
        if reduction == "sum":
            score = selected_metric.sum(dim=-1)
        elif reduction == "mean":
            score = selected_metric.mean(dim=-1)
        else:
            score = selected_metric

        return {
            "score_type": score_type,
            "reduction": reduction,
            "score": score,
            "target_token_indices": selected_indices,
            "target_token_ids": target_token_ids,
            "target_logits": target_logits,
            "target_logprobs": target_logprobs,
            "target_probabilities": target_probabilities,
            "logit_sum": target_logits.sum(dim=-1),
            "logit_mean": target_logits.mean(dim=-1),
            "logprob_sum": target_logprobs.sum(dim=-1),
            "logprob_mean": target_logprobs.mean(dim=-1),
            "probability_sum": target_probabilities.sum(dim=-1),
            "probability_mean": target_probabilities.mean(dim=-1),
            "prompt_lengths": prompt_lengths,
            "used_logits_to_keep": used_logits_to_keep,
        }

    def forward(self, input_ids, attention_mask=None):
        if self.prompt_length is None or self.generated_answer_ids is None:
            raise ValueError("Generation targets are not initialized.")
        if self.target_token_position is None or self.selected_interpretation_token_word_id is None:
            raise ValueError("Selected interpretation targets are not initialized.")

        single_input = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            single_input = True
        input_ids = input_ids.to(self.model.device, dtype=torch.long)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(self.model.device, dtype=torch.long)

        max_target_position = int(self.target_token_position.max().item())
        generated_prefix_len = max(0, max_target_position - self.prompt_length)
        generated_prefix = self.generated_answer_ids[:generated_prefix_len]
        if generated_prefix_len:
            prefix_batch = generated_prefix.unsqueeze(0).expand(input_ids.shape[0], -1)
            input_ids = torch.cat([input_ids, prefix_batch], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(prefix_batch, device=self.model.device)],
                dim=1,
            )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
            )

        logits = outputs.logits[:, self.target_token_position - 1]
        probabilities = self.softmax(logits)
        selected_token_ids = self.selected_interpretation_token_word_id.to(self.model.device)
        gather_indices = selected_token_ids.unsqueeze(0).unsqueeze(-1).expand(
            input_ids.shape[0], -1, -1
        )
        selected_probabilities = probabilities.gather(dim=2, index=gather_indices).squeeze(-1)
        return selected_probabilities[0] if single_input else selected_probabilities
