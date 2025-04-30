# Copyright © 2024 Apple Inc.

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ..models import cache
from ..generate import generate_step
from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_extract_xml_answer,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)
from .graph_reward_functions import (
    expert_reward_func,
    strict_format_reward_func,
    reward_len,
)
from .trainer import TrainingArgs, TrainingCallback, average_gradients, grad_checkpoint


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    epsilon_low: float = field(
        default=1e-4,
        metadata={"help": "The lower bound Epsilon for numerical stability."},
    )
    epsilon_high: float = field(
        default=2e-4,
        metadata={"help": "The upper bound Epsilon for numerical stability."},
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of Generations."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )
    num_iterations: int = field(
        default=4,
        metadata={
            "help": "Number of policy updates per batch generation. Values > 1 enable PPO clipping."
        },
    )
    enable_overlong_filtering: bool = field(
        default=False,
        metadata={
            "help": "Enable overlong filtering to mask loss for truncated sequences. This helps preserve long-context reasoning."
        },
    )


def get_per_token_logps(model: nn.Module, inputs, lengths):
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]
    per_token_logps = []
    for i in range(logits.shape[0]):
        seq_len = int(lengths[i]) - 1
        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits)
    return per_token_logps


def generate_grpo(
    model: nn.Module,
    tokenizer,
    prompt_tokens,
    max_tokens: int,
    group_size: int,
    temperature: float,
    batch_size: int,
    end_token: str = "</answer>",
    prompts_text: List[str] = [],
    answers_text: List[str] = [],
    util_valid_score: bool = False,
):
    try:
        end_sequence = mx.array(tokenizer.encode(end_token))
        total_samples = len(prompt_tokens)
        all_completions = []
        all_completion_texts = []
        batch_indices = []
        for i in range(0, total_samples, batch_size):
            current_batch_size = min(batch_size, total_samples - i)
            batch_prompts = prompt_tokens[i : i + current_batch_size]
            max_prompt_len = max(len(p) for p in batch_prompts)
            padded_prompts = []
            for prompt in batch_prompts:
                padding = [tokenizer.pad_token_id] * (max_prompt_len - len(prompt))
                padded_prompts.append(prompt + padding)

            prompt_tensor = mx.stop_gradient(mx.array(padded_prompts))
            if len(prompt_tensor.shape) == 1:
                prompt_tensor = prompt_tensor[None, :]
            if prompt_tensor.shape[1] == 0:
                continue

            expanded_prompts = mx.repeat(prompt_tensor, group_size, axis=0)
            batch_results = []
            total_prompt_samples = expanded_prompts.shape[0]
            for prompt_idx in range(0, total_prompt_samples, group_size):
                prompt_group_results = []
                prompt_group_texts = []
                prompt_batch_idx = i + prompt_idx // group_size

                # Generate the initial group of completions
                for g in range(group_size):
                    current_prompt_idx = prompt_idx + g
                    if current_prompt_idx >= total_prompt_samples:
                        break

                    current_tokens = []
                    prompt_cache = cache.make_prompt_cache(model)
                    for token, _ in generate_step(
                        expanded_prompts[current_prompt_idx],
                        model,
                        max_tokens=max_tokens,
                        sampler=lambda x: mx.random.categorical(x / temperature),
                        prompt_cache=prompt_cache,
                    ):
                        if token == tokenizer.eos_token_id:
                            break

                        current_tokens.append(token)
                        if len(current_tokens) >= len(end_sequence) and mx.array_equal(
                            mx.array(current_tokens[-len(end_sequence) :]), end_sequence
                        ):
                            break

                    if current_tokens:
                        prompt_group_results.append(mx.array(current_tokens))
                        completion_text = tokenizer.decode(current_tokens)
                        prompt_group_texts.append(completion_text)

                # Check if we need to validate scores
                valid_score_obtained = False
                additional_attempts = 0
                max_additional_attempts = 10

                if (
                    util_valid_score
                    and prompt_group_texts
                    and prompt_batch_idx < len(prompts_text)
                    and prompt_batch_idx < len(answers_text)
                ):
                    try:
                        # Check initial group of completions
                        scores = expert_reward_func(
                            prompts=[prompts_text[prompt_batch_idx]]
                            * len(prompt_group_texts),
                            completions=prompt_group_texts,
                            answer=[answers_text[prompt_batch_idx]]
                            * len(prompt_group_texts),
                        )

                        if scores and sum(scores) > 0:
                            valid_score_obtained = True

                        # If no valid scores, generate one at a time until we find one or reach max attempts
                        while (
                            not valid_score_obtained
                            and additional_attempts < max_additional_attempts
                        ):
                            print(
                                f"Generate another completion {additional_attempts} to build effective gradident...",
                                round(mx.get_peak_memory() / 1e9, 2),
                                flush=True,
                            )
                            additional_attempts += 1

                            # Generate one more completion
                            current_tokens = []
                            prompt_cache = cache.make_prompt_cache(model)
                            for token, _ in generate_step(
                                expanded_prompts[
                                    prompt_idx
                                ],  # Use the first prompt in the group
                                model,
                                max_tokens=max_tokens,
                                sampler=lambda x: mx.random.categorical(
                                    x / (temperature)
                                ),
                                prompt_cache=prompt_cache,
                            ):
                                if token == tokenizer.eos_token_id:
                                    break

                                current_tokens.append(token)
                                if len(current_tokens) >= len(
                                    end_sequence
                                ) and mx.array_equal(
                                    mx.array(current_tokens[-len(end_sequence) :]),
                                    end_sequence,
                                ):
                                    break

                            if current_tokens:
                                new_completion = mx.array(current_tokens)
                                new_completion_text = tokenizer.decode(current_tokens)

                                print(
                                    f"Generated another completion ...\n```{new_completion_text[-500:]}"
                                )
                                # Check if this single completion gets a positive score
                                single_score = expert_reward_func(
                                    prompts=[prompts_text[prompt_batch_idx]],
                                    completions=[new_completion_text],
                                    answer=[answers_text[prompt_batch_idx]],
                                )

                                if single_score and single_score[0] > 0:
                                    # Add this good completion to our results
                                    prompt_group_results.append(new_completion)
                                    prompt_group_texts.append(new_completion_text)
                                    # If we have more than group_size completions, pop the first one
                                    if len(prompt_group_results) > group_size:
                                        prompt_group_results.pop(0)
                                        prompt_group_texts.pop(0)
                                    valid_score_obtained = True
                                    break
                    except Exception as e:
                        print(f"Error evaluating scores: {e}")
                        valid_score_obtained = True  # Continue on error
                else:
                    valid_score_obtained = True  # Skip validation if not required

                # Add all results to our batch
                batch_results.extend(prompt_group_results)
                for completion_ids in prompt_group_results:
                    completion_text = tokenizer.decode(completion_ids.tolist())
                    all_completions.append(mx.stop_gradient(completion_ids))
                    all_completion_texts.append(completion_text)
                    batch_indices.append(prompt_batch_idx)

    finally:
        mx.clear_cache()

    return all_completions, all_completion_texts, batch_indices


def grpo_loss(
    model,
    ref_model,
    tokenizer,
    batch,
    completions=None,
    completion_texts=None,
    batch_indices=None,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    group_size: int = 4,
    epsilon_low: float = 1e-4,
    epsilon_high: float = 2e-4,
    max_tokens: int = 64,
    temperature: float = 0.8,
    reward_weights: Optional[List[float]] = None,
    batch_size: int = 1,
    is_validation: bool = False,
    old_log_probs=None,  # Add parameter for old_log_probs
    enable_overlong_filtering: bool = False,
):
    prompt_tokens, _, prompt_text, answer_text, type_info = batch

    if (
        completions is not None
        and completion_texts is not None
        and batch_indices is not None
    ):
        all_completions = completions
        all_completion_texts = completion_texts
        batch_indices = batch_indices
    else:
        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            group_size=group_size,
            temperature=temperature,
            batch_size=batch_size,
        )

    if not all_completions:
        raise ValueError(
            "No completions were generated. Please check your model and inputs."
        )

    expanded_answers = []
    expanded_prompts = []
    expanded_types = []
    unique_prompt_indices = sorted(set(batch_indices))
    grouped_completions = {idx: [] for idx in unique_prompt_indices}

    for i, completion_idx in enumerate(batch_indices):
        grouped_completions[completion_idx].append(i)

    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []

    for prompt_idx in unique_prompt_indices:
        completion_indices = grouped_completions[prompt_idx]
        for idx in completion_indices:
            ordered_completions.append(all_completions[idx])
            ordered_completion_texts.append(all_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)
            expanded_answers.append(answer_text[prompt_idx])
            expanded_prompts.append(prompt_text[prompt_idx])
            expanded_types.append(
                type_info[prompt_idx] if type_info is not None else None
            )

    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices
    max_length = max(ids.shape[0] for ids in all_completions)
    padded_completions = []
    attention_masks = []

    for completion_ids in all_completions:
        completion_tensor = mx.array(completion_ids.tolist())
        padding_length = max_length - completion_tensor.shape[0]
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
            padded_ids = mx.concatenate([completion_tensor, padding])
            mask = mx.concatenate(
                [mx.ones_like(completion_tensor), mx.zeros_like(padding)]
            )
        else:
            padded_ids = completion_tensor
            mask = mx.ones_like(completion_tensor)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs = mx.stack(padded_completions)
    attention_mask = mx.stack(attention_masks)
    lengths = attention_mask.sum(axis=1)

    token_log_probs = get_per_token_logps(model, inputs, lengths)
    mx.eval(token_log_probs)

    print("get model logps", round(mx.get_peak_memory() / 1e9, 2))

    # Use old_log_probs if provided, otherwise default to current token_log_probs
    # This is equivalent to policy_ratio=1 when old_log_probs is None
    log_probs_old = old_log_probs if old_log_probs is not None else token_log_probs

    # Calculate reference model log probs for KL divergence (separate from old_log_probs)
    if ref_model is None:
        ref_token_log_probs = token_log_probs  # Fallback if no ref_model
    else:
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths)
        mx.eval(ref_token_log_probs)

    # Pad and stack all three lists consistently
    max_len = max(x.shape[0] for x in token_log_probs)
    padded_log_probs = []
    padded_ref_log_probs = []
    padded_old_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))

        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

        # Pad old_log_probs
        old_len = log_probs_old[i].shape[0] if i < len(log_probs_old) else 0
        old_padding = (
            mx.zeros((max_len - old_len,)) if old_len > 0 else mx.zeros((max_len,))
        )
        if old_len > 0:
            padded_old_log_probs.append(mx.concatenate([log_probs_old[i], old_padding]))
        else:
            padded_old_log_probs.append(
                mx.concatenate([token_log_probs[i], old_padding])
            )

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)
    old_log_probs = mx.stack(padded_old_log_probs)

    all_func_rewards = []
    for reward_func in reward_funcs:
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
            types=expanded_types,
        )
        if raw_rewards is None:
            processed_rewards = [float("nan")] * len(all_completion_texts)
        else:
            processed_rewards = [
                float(r) if r is not None else float("nan") for r in raw_rewards
            ]
        func_rewards = mx.array(processed_rewards)
        all_func_rewards.append(func_rewards)

    rewards = mx.stack(all_func_rewards, axis=1)

    all_nan_rows = mx.all(mx.isnan(rewards), axis=1)
    if mx.any(all_nan_rows):
        nan_row_idx = mx.argmax(all_nan_rows).item()
        warning_msg = (
            f"All reward functions returned None for prompt: {expanded_prompts[nan_row_idx]}, "
            f"completion: {all_completion_texts[nan_row_idx]}, "
            f"answer: {expanded_answers[nan_row_idx]}. "
            "Please ensure that at least one reward function returns a valid reward."
        )
        raise RuntimeError(warning_msg)

    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)

    valid_reward_mask = ~mx.isnan(rewards)
    rewards_no_nan = mx.where(valid_reward_mask, rewards, mx.zeros_like(rewards))
    rewards = (rewards_no_nan * mx.expand_dims(reward_weights, 0)).sum(axis=1)

    num_unique_prompts = len(unique_prompt_indices)

    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:
            prompt_rewards = mx.array(prompt_rewards)
            mean_reward = mx.mean(prompt_rewards)
            std_reward = mx.std(prompt_rewards)
            indices = [
                j
                for j, idx in enumerate(batch_indices)
                if idx == unique_prompt_indices[i]
            ]
            for j, idx in enumerate(indices):
                advantages[idx] = (prompt_rewards[j] - mean_reward) / (
                    std_reward + 1e-6
                )
                print(f"compute advantages({idx}) = {advantages[idx]}")
        else:
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    # Compute KL divergence using Schulman's approximator
    kl_div = (
        mx.exp(ref_token_log_probs - token_log_probs)
        - (ref_token_log_probs - token_log_probs)
        - 1
    )

    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Implement overlong filtering logic
    if enable_overlong_filtering:
        print("enable_overlong_filtering", flush=True)
        # Identify truncated sequences (those reaching max_tokens)
        is_truncated = []
        # Check if any tokens represent an end token in the vocabulary
        eos_id = tokenizer.eos_token_id
        # Try to encode "</answer>" as a common end token if it exists in the vocabulary
        try:
            end_token_ids = tokenizer.encode("</answer>")
            has_end_token = len(end_token_ids) > 0
        except:
            has_end_token = False
            end_token_ids = []

        for i, completion_ids in enumerate(all_completions):
            # A sequence is considered truncated if it reaches max_tokens
            # and doesn't end with an end token or EOS token
            if len(completion_ids) >= max_tokens:
                # Check if the sequence ends with EOS
                if completion_ids[-1] == eos_id:
                    is_truncated.append(False)
                    continue

                # Check if it ends with the end token sequence
                if has_end_token and len(completion_ids) >= len(end_token_ids):
                    last_tokens = completion_ids[-len(end_token_ids) :]
                    if mx.array_equal(mx.array(last_tokens), mx.array(end_token_ids)):
                        is_truncated.append(False)
                        continue

                # If we get here, the sequence was truncated
                is_truncated.append(True)
                print(f"truncated sequence index {i}", flush=True)
            else:
                is_truncated.append(False)

        is_truncated = mx.array(is_truncated)

        # For truncated sequences, completely mask all tokens
        if mx.any(is_truncated):
            # Create a mask for truncated sequences
            truncation_mask = mx.ones_like(length_mask)
            for i, truncated in enumerate(is_truncated):
                if truncated:
                    # Mask all tokens for truncated sequences
                    seq_len = int(lengths[i]) - 1
                    truncation_mask[i, :seq_len] = 0

            # Apply truncation mask to the length mask
            length_mask = length_mask * truncation_mask

            # Log how many sequences were truncated
            truncated_count = mx.sum(is_truncated)
            if is_validation:
                print(
                    f"Overlong filtering applied to {truncated_count} out of {len(is_truncated)} sequences"
                )

    # Compute correct policy ratio using old_log_probs
    policy_ratio = mx.exp(mx.array(token_log_probs - old_log_probs))
    print(f"compute policy_ratio = {mx.mean(policy_ratio)}", flush=True)

    # Apply asymmetric PPO clipping instead of symmetric clipping
    # This handles positive and negative advantages differently
    # and is more effective in practice
    policy_ratio_cliped = mx.clip(policy_ratio, 1 - epsilon_low, 1 + epsilon_high)

    # Track clipping metrics
    is_low_clipped = (policy_ratio < 1 - epsilon_low) & (advantages.reshape(-1, 1) < 0)
    is_high_clipped = (policy_ratio > 1 + epsilon_high) & (
        advantages.reshape(-1, 1) > 0
    )
    is_region_clipped = is_low_clipped | is_high_clipped

    # Calculate both unclipped and clipped objectives
    unclipped_obj = policy_ratio * advantages.reshape(-1, 1)
    clipped_obj = policy_ratio_cliped * advantages.reshape(-1, 1)

    # Take the minimum (pessimistic bound)
    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)

    # Add KL penalty if beta is non-zero
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_div

    # Average over tokens
    loss = (per_token_loss * length_mask).sum() / length_mask.sum()

    # Calculate mean KL divergence for metrics
    mean_kl = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()

    reward_metrics = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        valid_mask = ~mx.isnan(
            mx.array(
                [
                    reward if reward is not None else float("nan")
                    for reward in raw_rewards
                ]
            )
        )
        valid_rewards = mx.array(
            [
                reward
                for reward in raw_rewards
                if reward is not None and not mx.isnan(reward)
            ]
        )
        if len(valid_rewards) > 0:
            reward_metrics[f"{func_name}_mean"] = mx.mean(valid_rewards)
            reward_metrics[f"{func_name}_std"] = (
                mx.std(valid_rewards) if len(valid_rewards) > 1 else mx.zeros(1)
            )
            reward_metrics[f"{func_name}_coverage"] = valid_mask.sum() / len(
                raw_rewards
            )
        else:
            reward_metrics[f"{func_name}_mean"] = float("nan")
            reward_metrics[f"{func_name}_std"] = float("nan")
            reward_metrics[f"{func_name}_coverage"] = 0.0

    grouped_rewards_mean = mx.array(
        [mx.mean(mx.array(rewards)) for rewards in rewards_by_prompt]
    )
    grouped_rewards_std = mx.array(
        [
            mx.std(mx.array(rewards)) if len(rewards) > 1 else mx.zeros(1)
            for rewards in rewards_by_prompt
        ]
    )

    metrics = {
        "total_rewards_mean": mx.mean(rewards),
        "total_rewards_std": mx.std(rewards),
        "grouped_rewards_mean": mx.mean(grouped_rewards_mean),
        "grouped_rewards_std": mx.mean(grouped_rewards_std),
        "kl": mean_kl,
        "average_generated_tokens": sum([len(c) for c in all_completions])
        // len(batch_indices),
        "clip_ratio_low": (
            (is_low_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        "clip_ratio_high": (
            (is_high_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        "clip_ratio_total": (
            (is_region_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        **reward_metrics,
    }

    if is_validation and all_completion_texts:
        # if all_completion_texts:
        print("\n=== Validation Sample Details ===")
        last_prompt_idx = batch_indices[-1] if batch_indices else 0
        if last_prompt_idx < len(prompt_text):
            print(f"\n📋 Raw Prompt:\n{prompt_text[last_prompt_idx]}")
            print("\n" + "=" * 10 + "\n")
            if last_prompt_idx < len(prompt_tokens):
                actual_prompt = tokenizer.decode(prompt_tokens[last_prompt_idx])
                print(f"\n🔄 Model Input:\n{actual_prompt}")
                print("\n" + "=" * 10 + "\n")

        for index, completion_text in enumerate(all_completion_texts):
            print(f"\n📝 Generation {index+1}:\n{completion_text}")
            print("\n" + "=" * 10 + "\n")

        print(f"\n💭 Metrics {metrics}")

        if last_prompt_idx < len(answer_text):
            print(f"\n✅ Answer:\n{answer_text[last_prompt_idx]}")
            print("\n" + "=" * 10 + "\n")
        """
        if "r1_extract_xml_answer" in globals():
            print(
                f"\n🔍 Extracted Answer:\n{r1_extract_xml_answer(all_completion_texts[-1])}"
            )
        """
        print("\n" + "=" * 35 + "\n")

    mx.clear_cache()

    return loss, length_mask.sum(axis=1).sum(), metrics


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    has_types = isinstance(dataset[0], tuple) and len(dataset[0]) == 5

    if (
        not dataset
        or not isinstance(dataset[0], tuple)
        or (not has_types and len(dataset[0]) != 4)
    ):
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str[, type]) tuples"
        )

    def length_key(i):
        return len(dataset[i][0]) + len(dataset[i][1])

    idx = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
            current_batch = [dataset[j] for j in batch_idx]

            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]
            types = [item[4] for item in current_batch] if has_types else None

            if any(len(p) > max_seq_length for p in prompts_tokens):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )

            yield prompts_tokens, answers_tokens, prompts_text, answers_text, types

        if not train:
            break


def evaluate_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    reward_funcs: Optional[List[RewardFunctions]] = [
        expert_reward_func,
    ],
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    enable_overlong_filtering: bool = False,
):
    all_losses = 0
    ntokens = 0
    all_metrics = None

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks, metrics = loss_fn(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            ref_model=ref_model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_validation=True,
            old_log_probs=None,
            reward_weights=None,
            enable_overlong_filtering=enable_overlong_filtering,
        )

        all_losses += losses * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, ntokens, avg_metrics


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = [
        expert_reward_func,
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
):
    print(
        f"Starting GRPO training with {len(reward_funcs)} reward functions..., "
        f"iters: {args.iters}, num_iterations: {args.num_iterations}, "
        f"beta: {args.beta}, group_size: {args.group_size}, "
        f"epsilon: {args.epsilon_low}, epsilon_high: {args.epsilon_high}, "
        f"temperature: {args.temperature}, max_tokens: {args.max_completion_length}, "
        f"overlong_filtering: {args.enable_overlong_filtering}"
    )
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    def step(batch, outer_it):
        prompt_tokens, targets, prompt_lens, target_lens, type_info = batch
        total_loss = 0
        total_tokens = 0
        total_metrics = {}

        # Generate completions once for this batch
        print(
            f"Start training on batch [Outer Iteration {outer_it}]  - {type_info}",
            flush=True,
        )
        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            temperature=args.temperature,
            batch_size=args.batch_size,
            prompts_text=prompt_lens,
            answers_text=target_lens,
            util_valid_score=True,
        )

        # Prepare inputs for log probability calculation
        max_length = max(ids.shape[0] for ids in all_completions)
        padded_completions = []
        attention_masks = []

        for completion_ids in all_completions:
            completion_tensor = mx.array(completion_ids.tolist())
            padding_length = max_length - completion_tensor.shape[0]
            if padding_length > 0:
                padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
                padded_ids = mx.concatenate([completion_tensor, padding])
                mask = mx.concatenate(
                    [mx.ones_like(completion_tensor), mx.zeros_like(padding)]
                )
            else:
                padded_ids = completion_tensor
                mask = mx.ones_like(completion_tensor)
            padded_completions.append(padded_ids)
            attention_masks.append(mask)

        inputs = mx.stack(padded_completions)
        attention_mask = mx.stack(attention_masks)
        lengths = attention_mask.sum(axis=1)

        # Calculate log_probs using the current model (before any updates)
        token_log_probs = get_per_token_logps(model, inputs, lengths)
        # Apply stop_gradient to each array in the list
        old_log_probs = [mx.stop_gradient(log_prob) for log_prob in token_log_probs]
        mx.eval(old_log_probs)

        print("compute old_log_probs", round(mx.get_peak_memory() / 1e9, 2))

        # Perform multiple updates on the same batch data
        for i in range(args.num_iterations):
            print(f"Update {outer_it} (Inner {i+1}/{args.num_iterations})", flush=True)

            # Compute loss and gradients using fixed old_log_probs
            (loss, toks, metrics), grad = loss_value_and_grad(
                model,
                tokenizer=tokenizer,
                batch=(prompt_tokens, targets, prompt_lens, target_lens, type_info),
                completions=all_completions,
                completion_texts=all_completion_texts,
                batch_indices=batch_indices,
                reward_funcs=reward_funcs,
                beta=args.beta,
                group_size=args.group_size,
                epsilon_low=args.epsilon_low,
                epsilon_high=args.epsilon_high,
                ref_model=ref_model,
                old_log_probs=old_log_probs,
                reward_weights=args.reward_weights,
                enable_overlong_filtering=args.enable_overlong_filtering,
            )
            print("compute loss_value_and_grad", round(mx.get_peak_memory() / 1e9, 2))

            # Update model parameters
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            print("compute optimizer.update", round(mx.get_peak_memory() / 1e9, 2))
            # Print the loss for this inner update
            print(
                f"Update {outer_it} (Inner {i+1}/{args.num_iterations}) - Loss: {loss:.4f}",
                flush=True,
            )

            # Accumulate loss and metrics
            total_loss += loss
            total_tokens += toks

            # Initialize or accumulate metrics
            if not total_metrics:
                total_metrics = {k: v for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    total_metrics[k] += v

            # Clear cache after each update
            mx.clear_cache()
            mx.eval(state)

        # Average the accumulated metrics over num_iterations
        avg_metrics = {k: v / args.num_iterations for k, v in total_metrics.items()}
        avg_loss = total_loss / args.num_iterations

        print(
            f"[Outer Iteration {outer_it}] Completed {args.num_iterations} updates - Avg Loss: {avg_loss:.4f}",
            flush=True,
        )

        return total_loss, total_tokens, avg_metrics

    # Initialize metrics tracking
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "total_rewards_mean": 0,
        "total_rewards_std": 0,
        "grouped_rewards_mean": 0,
        "grouped_rewards_std": 0,
        "kl": 0,
        "average_generated_tokens": 0,
        "clip_ratio_low": 0,
        "clip_ratio_high": 0,
        "clip_ratio_total": 0,
    }
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"{func_name}_mean"] = 0
        accumulated_metrics[f"{func_name}_std"] = 0
        accumulated_metrics[f"{func_name}_coverage"] = 0

    start = time.perf_counter()

    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        # Evaluation logic
        if it == 1 or it % args.steps_per_eval == 0 or it >= args.iters:
            stop = time.perf_counter()
            val_loss, val_ntokens, val_metrics = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                tokenizer=tokenizer,
                group_size=3,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                beta=args.beta,
                epsilon_low=args.epsilon_low,
                epsilon_high=args.epsilon_high,
                temperature=args.temperature,
                iterate_batches=iterate_batches,
                enable_overlong_filtering=args.enable_overlong_filtering,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                val_metrics_str = (
                    f"Val loss {val_loss:.3f}, "
                    f"Val total_rewards_mean {val_metrics['total_rewards_mean']:.3f}, "
                    f"Val total_rewards_std {val_metrics['total_rewards_std']:.3f}, "
                    f"Val grouped_rewards_mean {val_metrics['grouped_rewards_mean']:.3f}, "
                    f"Val grouped_rewards_std {val_metrics['grouped_rewards_std']:.3f}, "
                    f"Val Average Generated Tokens {val_metrics['average_generated_tokens']}, "
                    f"Val kl {val_metrics['kl']:.3f}"
                )

                for i, reward_func in enumerate(reward_funcs):
                    val_metrics_str += (
                        f", Val {reward_func.__name__}_mean {val_metrics[f'{reward_func.__name__}_mean']:.3f}, "
                        f"Val {reward_func.__name__}_std {val_metrics[f'{reward_func.__name__}_std']:.3f}"
                    )

                print(
                    f"Iter {it}: {val_metrics_str}, " f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

            start = time.perf_counter()

        # Call step() which now internally performs num_iterations updates
        loss, toks, metrics = step(batch, it)
        losses += loss
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                train_metrics_str = (
                    f"Train loss {float(train_loss):.3f}, "
                    f"Total rewards mean {float(avg_metrics['total_rewards_mean']):.3f}, "
                    f"Total rewards std {float(avg_metrics['total_rewards_std']):.3f}, "
                    f"Grouped rewards mean {float(avg_metrics['grouped_rewards_mean']):.3f}, "
                    f"Grouped rewards std {float(avg_metrics['grouped_rewards_std']):.3f}, "
                    f"Average Generated Tokens {float(avg_metrics['average_generated_tokens'])}, "
                    f"KL {float(avg_metrics['kl']):.3f}"
                )

                for i, reward_func in enumerate(reward_funcs):
                    func_name = reward_func.__name__
                    train_metrics_str += (
                        f", {func_name} mean {float(avg_metrics[f'{func_name}_mean']):.3f}, "
                        f"{func_name} std {float(avg_metrics[f'{func_name}_std']):.3f}"
                    )

                print(
                    f"Iter {it}: {train_metrics_str}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": it,
                        "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in avg_metrics.items()},
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem,
                    }
                )

            losses = 0
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Final save
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
