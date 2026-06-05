from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    tokenized: list[tuple[list[int], list[int], list[int]]] = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        full_ids = prompt_ids + output_ids
        if len(full_ids) < 2:
            raise ValueError("prompt + output must contain at least two tokens")
        tokenized.append((prompt_ids, output_ids, full_ids))
        max_len = max(max_len, len(full_ids) - 1)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((len(tokenized), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(tokenized), max_len), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((len(tokenized), max_len), dtype=torch.bool)

    for row, (prompt_ids, _, full_ids) in enumerate(tokenized):
        row_input = full_ids[:max_len]
        row_labels = full_ids[1 : max_len + 1]
        length = len(row_input)
        input_ids[row, :length] = torch.tensor(row_input, dtype=torch.long)
        labels[row, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
        prompt_len = len(prompt_ids)
        response_mask[row, : len(row_labels)] = torch.tensor(
            [(label_position + 1) >= prompt_len for label_position in range(len(row_labels))],
            dtype=torch.bool,
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    logits = model(input_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    labels = labels.to(input_ids.device)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = {"log_probs": log_probs}
    if return_token_entropy:
        probs = log_probs_all.exp()
        output["token_entropy"] = -(probs * log_probs_all).sum(dim=-1)
    return output


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor([reward["reward"] for reward in reward_dicts], dtype=torch.float32)
    metadata = {
        "reward_mean": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
        "format_reward_mean": float(torch.tensor([r.get("format_reward", 0.0) for r in reward_dicts]).mean().item())
        if reward_dicts
        else 0.0,
        "answer_reward_mean": float(torch.tensor([r.get("answer_reward", 0.0) for r in reward_dicts]).mean().item())
        if reward_dicts
        else 0.0,
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    flat_rewards = raw_rewards.float().reshape(-1)
    if flat_rewards.numel() % group_size != 0:
        raise ValueError("raw_rewards length must be divisible by group_size")
    grouped = flat_rewards.reshape(-1, group_size)

    if baseline == "mean":
        centered = grouped - grouped.mean(dim=1, keepdim=True)
    elif baseline == "none":
        centered = grouped
    else:
        raise ValueError(f"unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        advantages = centered / (grouped.std(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "none":
        advantages = centered
    elif advantage_normalizer == "mean":
        advantages = centered / (grouped.mean(dim=1, keepdim=True) + advantage_eps)
    else:
        raise ValueError(f"unsupported advantage_normalizer: {advantage_normalizer}")

    metadata = {
        "reward_mean": float(flat_rewards.mean().item()) if flat_rewards.numel() else 0.0,
        "reward_std": float(flat_rewards.std().item()) if flat_rewards.numel() > 1 else 0.0,
        "reward_min": float(flat_rewards.min().item()) if flat_rewards.numel() else 0.0,
        "reward_max": float(flat_rewards.max().item()) if flat_rewards.numel() else 0.0,
    }
    return advantages.reshape_as(flat_rewards), metadata


def _advantages_column(raw_rewards_or_advantages: Tensor, like: Tensor) -> Tensor:
    advantages = raw_rewards_or_advantages.to(device=like.device, dtype=like.dtype)
    return advantages.reshape(advantages.shape[0], 1)


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    advantages = _advantages_column(raw_rewards_or_advantages, policy_log_probs)
    metadata: dict[str, Tensor] = {}

    if importance_reweighting_method == "none":
        return -advantages * policy_log_probs, metadata

    if old_log_probs is None:
        raise ValueError("old_log_probs is required for off-policy losses")
    log_ratio = policy_log_probs - old_log_probs.to(policy_log_probs.device)

    if importance_reweighting_method == "noclip":
        ratio = torch.exp(log_ratio)
        return -advantages * ratio, metadata

    if cliprange is None:
        raise ValueError("cliprange is required for clipped losses")

    if importance_reweighting_method == "grpo":
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        objective = torch.minimum(advantages * ratio, advantages * clipped_ratio)
        metadata["clip_fraction"] = (ratio.ne(clipped_ratio)).float().mean()
        return -objective, metadata

    if importance_reweighting_method == "gspo":
        if response_mask is None:
            response_mask = torch.ones_like(policy_log_probs, dtype=torch.bool)
        response_mask = response_mask.to(policy_log_probs.device).bool()
        denom = response_mask.sum(dim=1, keepdim=True).clamp_min(1)
        sequence_log_ratio = (log_ratio * response_mask).sum(dim=1, keepdim=True) / denom
        sequence_ratio = torch.exp(sequence_log_ratio)
        clipped_ratio = torch.clamp(sequence_ratio, 1.0 - cliprange, 1.0 + cliprange)
        objective = torch.minimum(advantages * sequence_ratio, advantages * clipped_ratio)
        metadata["clip_fraction"] = (sequence_ratio.ne(clipped_ratio)).float().mean()
        return -objective.expand_as(policy_log_probs), metadata

    raise ValueError(f"unsupported importance_reweighting_method: {importance_reweighting_method}")


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    mask = mask.to(device=per_token_policy_gradient_loss.device, dtype=per_token_policy_gradient_loss.dtype)
    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        per_sequence = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return per_sequence.mean()
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for constant normalization")
        return masked_loss.sum() / normalization_constant
    raise ValueError(f"unsupported loss_normalization: {loss_normalization}")


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    model.train()
    device = next(model.parameters()).device
    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    batch_size = input_ids.shape[0]
    optimizer.zero_grad(set_to_none=True)
    total_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
    entropy_numer = input_ids.new_tensor(0.0, dtype=torch.float32)
    entropy_denom = input_ids.new_tensor(0.0, dtype=torch.float32)

    indices = torch.arange(batch_size, device=device)
    chunks = torch.chunk(indices, gradient_accumulation_steps)
    for chunk in chunks:
        if chunk.numel() == 0:
            continue
        result = get_response_log_probs(
            model=model,
            input_ids=input_ids[chunk],
            labels=labels[chunk],
            return_token_entropy=True,
        )
        micro_old_log_probs = old_log_probs[chunk] if old_log_probs is not None else None
        per_token_loss, _ = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages[chunk],
            policy_log_probs=result["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=micro_old_log_probs,
            cliprange=cliprange,
            response_mask=response_mask[chunk],
        )
        micro_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=response_mask[chunk],
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        if loss_normalization == "sequence":
            backward_loss = micro_loss * (chunk.numel() / batch_size)
        else:
            backward_loss = micro_loss
        backward_loss.backward()
        total_loss = total_loss + backward_loss.detach()
        entropy_numer = entropy_numer + (result["token_entropy"] * response_mask[chunk]).sum().detach()
        entropy_denom = entropy_denom + response_mask[chunk].sum().detach()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = _compute_grad_norm(model.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata: dict[str, Tensor | float] = {
        "loss": float(total_loss.item()),
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm),
        "token_entropy": float((entropy_numer / entropy_denom.clamp_min(1)).item()),
        **reward_metadata,
        **advantage_metadata,
    }
    return total_loss.detach(), metadata


def _compute_grad_norm(parameters) -> Tensor:
    grads = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(grads), ord=2)


class PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.examples[index]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    prompt_template = (Path(__file__).resolve().parent / "prompts_safety/alpaca_sft.prompt").read_text()
    rows = [json.loads(line) for line in Path(dataset_path).read_text().splitlines() if line.strip()]
    if shuffle:
        rng = random.Random(0)
        rng.shuffle(rows)

    token_ids: list[int] = []
    for row in rows:
        text = prompt_template.format(instruction=row["prompt"], response=row["response"]).rstrip()
        token_ids.extend(tokenizer.encode(text, add_special_tokens=True))
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)

    examples = []
    for start in range(0, len(token_ids) - seq_length, seq_length):
        chunk = token_ids[start : start + seq_length + 1]
        if len(chunk) != seq_length + 1:
            break
        examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )
    return PackedSFTDataset(examples)


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    del mmlu_example
    match = re.search(r"\b(?:answer\s+is|answer:|option)\s*([ABCD])\b", model_output, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", model_output)
    return match.group(1).upper() if match else None


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _response_log_prob_sum(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> Tensor:
    batch = tokenize_prompt_and_output([prompt], [response], tokenizer)
    input_ids = batch["input_ids"].to(next(model.parameters()).device)
    labels = batch["labels"].to(input_ids.device)
    response_mask = batch["response_mask"].to(input_ids.device)
    output = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
    return (output["log_probs"] * response_mask).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    prompt_template = (Path(__file__).resolve().parent / "prompts_safety/alpaca_sft.prompt").read_text()
    dpo_prompt = prompt_template.format(instruction=prompt, response="")
    eos = tokenizer.eos_token or ""
    chosen_response = response_chosen.rstrip() + eos
    rejected_response = response_rejected.rstrip() + eos

    chosen_logp = _response_log_prob_sum(lm, tokenizer, dpo_prompt, chosen_response)
    rejected_logp = _response_log_prob_sum(lm, tokenizer, dpo_prompt, rejected_response)
    with torch.no_grad():
        chosen_ref_logp = _response_log_prob_sum(lm_ref, tokenizer, dpo_prompt, chosen_response)
        rejected_ref_logp = _response_log_prob_sum(lm_ref, tokenizer, dpo_prompt, rejected_response)
    preference_logit = beta * ((chosen_logp - rejected_logp) - (chosen_ref_logp - rejected_ref_logp))
    return -F.logsigmoid(preference_logit)
