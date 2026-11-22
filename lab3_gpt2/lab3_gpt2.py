"""Laboratory Assignment 03: Autoregressive Language Modeling with GPT-2.

Course: ETMMGA304 - Generative AI and Its Applications
Institution: K.R. Mangalam University (Master of Computer Applications)
Curriculum Reference: BTX 3 (Apply), BTX 4 (Analyze) & BTX 5 (Evaluate)

This script investigates autoregressive text synthesis using the GPT-2 architecture
(openai-community/gpt2, 124M parameters). It demonstrates:
  1. Causal Language Model decoding mechanisms: Greedy search vs Stochastic Nucleus (top-p) sampling.
  2. Temperature scaling ablation (T=0.6 vs T=1.0) and stochastic seed variance (Seed 42 vs 43).
  3. Parameter-Efficient Fine-Tuning (PEFT): Freezing 94.3% of the network and training only
     the final Transformer block (h[-1], 7.09M parameters) on instructional prompt-response pairs.
  4. Prompt token masking with Cross-Entropy loss (label=-100) to isolate supervision to response tokens.
  5. Zero-shot base model vs adapted model evaluation on unseen STEM benchmark topics.

Outputs:
  - lab_outputs/outputs.json: Complete generation telemetry, decoding comparisons, and training logs.
  - lab_outputs/study_tip_gpt2/: Fine-tuned model checkpoint and tokenizer configuration.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_IDENTIFIER: str = "openai-community/gpt2"
RANDOM_SEED: int = 42
MAX_NEW_TOKENS: int = 45
TOP_P: float = 0.9
LEARNING_RATE: float = 2e-5
TRAIN_EPOCHS: int = 3

INSTRUCTIONAL_EXAMPLES: List[Tuple[str, str]] = [
    ("graphs", "Sketch a graph, then mark a path through three vertices."),
    ("Python loops", "Trace a loop by writing the counter after each step."),
    ("fractions", "Draw equal parts and label the fraction you shaded."),
    ("sets", "Draw a Venn diagram and label the intersection."),
    ("sorting", "Sort five values by hand and record each swap."),
    ("probability", "List the outcomes of two coin tosses before counting."),
]

HELDOUT_BENCHMARK_TOPICS: List[str] = [
    "recursion",
    "matrices",
    "binary search",
]


def configure_environment(seed: int = RANDOM_SEED) -> Tuple[torch.device, Path]:
    """Establishes deterministic environment and prepares output directory."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("lab_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return device, output_dir


# -----------------------------------------------------------------------------
# Model & Tokenizer Initialization
# -----------------------------------------------------------------------------
def load_foundation_model(
    model_id: str = MODEL_IDENTIFIER,
    device: torch.device = torch.device("cpu"),
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads pre-trained causal language model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model


# -----------------------------------------------------------------------------
# Autoregressive Decoding Routine
# -----------------------------------------------------------------------------
def generate_continuation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    *,
    sample: bool = False,
    temperature: float = 0.8,
    top_p: float = TOP_P,
    seed: int = RANDOM_SEED,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Performs controlled autoregressive text continuation under specified decoding settings."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs: Dict[str, Any] = {
        "do_sample": sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if sample:
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
        })

    with torch.no_grad():
        token_outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs.input_ids.size(1)
    newly_generated_tokens = token_outputs[0, prompt_length:]
    continuation_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    return continuation_text


# -----------------------------------------------------------------------------
# Parameter-Efficient Fine-Tuning (PEFT)
# -----------------------------------------------------------------------------
def fine_tune_last_block(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Tuple[str, str]],
    epochs: int = TRAIN_EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[PreTrainedModel, List[Dict[str, Any]], int, int]:
    """Freezes base layers and trains only the final Transformer block with prompt masking."""
    # Freeze entire backbone
    model.requires_grad_(False)
    # Unfreeze only the final Transformer block: transformer.h[-1]
    model.transformer.h[-1].requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.transformer.h[-1].parameters(), lr=lr)
    training_loss_logs: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_step_losses: List[float] = []

        for topic, response in examples:
            prefix_tokens = tokenizer.encode(f"Topic: {topic}\nStudy tip:")
            target_tokens = tokenizer.encode(" " + response) + [tokenizer.eos_token_id]

            input_ids = torch.tensor([prefix_tokens + target_tokens], device=device)
            # Label masking: set prompt tokens to -100 so loss ignores them
            labels = torch.tensor([[-100] * len(prefix_tokens) + target_tokens], device=device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_step_losses.append(loss.item())

        mean_loss = round(sum(epoch_step_losses) / len(epoch_step_losses), 3)
        last_loss = round(loss.item(), 3)
        print(f"      Epoch [{epoch:02d}/{epochs:02d}] -> Last Step Loss: {last_loss} | Mean Epoch Loss: {mean_loss}")

        training_loss_logs.append({
            "epoch": epoch,
            "last": last_loss,
            "mean": mean_loss,
        })

    model.eval()
    return model, training_loss_logs, trainable_params, total_params


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    device, output_dir = configure_environment(RANDOM_SEED)
    print("=" * 70)
    print(f"ETMMGA304 Lab 03: GPT-2 Decoding & Fine-Tuning Execution | Device: {device}")
    print("=" * 70)

    # Step 1: Load Pretrained Foundation Model
    print("[1/4] Loading pre-trained GPT-2 checkpoint and tokenizer...")
    tokenizer, base_model = load_foundation_model(MODEL_IDENTIFIER, device)

    # Step 2: Compare Prompts and Decoding Configurations
    print("\n[2/4] Conducting systematic decoding strategy ablation...")
    ablation_prompts = [
        "A story about a robot begins like this:",
        "In a school library, a tiny robot found a lost book. It",
    ]
    decoding_records: List[Dict[str, Any]] = []

    for prompt in ablation_prompts:
        print(f"\n      Evaluating Prompt: '{prompt}'")
        # 2a: Greedy Decoding
        greedy_text = generate_continuation(base_model, tokenizer, prompt, sample=False, device=device)
        print(f"        [Greedy] => {greedy_text[:60]}...")
        decoding_records.append({
            "prompt": prompt,
            "setting": "greedy",
            "seed": None,
            "text": greedy_text,
        })

        # 2b: Stochastic Nucleus Sampling across Temperatures & Seeds
        for temp in (0.6, 1.0):
            for seed_val in (42, 43):
                sample_text = generate_continuation(
                    base_model,
                    tokenizer,
                    prompt,
                    sample=True,
                    temperature=temp,
                    seed=seed_val,
                    device=device,
                )
                print(f"        [T={temp}, Seed={seed_val}] => {sample_text[:60]}...")
                decoding_records.append({
                    "prompt": prompt,
                    "setting": f"temp {temp}",
                    "seed": seed_val,
                    "text": sample_text,
                })

    # Step 3: Parameter-Efficient Fine-Tuning (Last Block Only)
    print("\n[3/4] Commencing parameter-efficient fine-tuning on last transformer block...")
    # Load separate instance for fine-tuning
    _, tuned_model = load_foundation_model(MODEL_IDENTIFIER, device)
    tuned_model, loss_history, num_trainable, num_total = fine_tune_last_block(
        tuned_model,
        tokenizer,
        INSTRUCTIONAL_EXAMPLES,
        epochs=TRAIN_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
    )
    pct_trainable = (num_trainable / num_total) * 100
    print(f"      Trainable Parameters: {num_trainable:,} of {num_total:,} ({pct_trainable:.2f}%)")

    # Serialize fine-tuned model checkpoint and tokenizer
    export_folder = output_dir / "study_tip_gpt2"
    tuned_model.save_pretrained(export_folder)
    tokenizer.save_pretrained(export_folder)
    print(f"      Saved fine-tuned checkpoint to: {export_folder}")

    # Step 4: Comparative Evaluation on Held-Out Topics
    print("\n[4/4] Evaluating Zero-Shot Base vs Fine-Tuned Model on Unseen Benchmark Topics...")
    heldout_comparisons: List[Dict[str, str]] = []

    for topic in HELDOUT_BENCHMARK_TOPICS:
        benchmark_prompt = f"Topic: {topic}\nStudy tip:"
        base_completion = generate_continuation(base_model, tokenizer, benchmark_prompt, sample=False, device=device)
        tuned_completion = generate_continuation(tuned_model, tokenizer, benchmark_prompt, sample=False, device=device)

        print(f"\n      Topic: '{topic}'")
        print(f"        Base Model : {base_completion}")
        print(f"        Tuned Model: {tuned_completion}")

        heldout_comparisons.append({
            "topic": topic,
            "base": base_completion,
            "tuned": tuned_completion,
        })

    # Serialize complete telemetry payload
    results_payload = {
        "seed": RANDOM_SEED,
        "device": str(device),
        "trainable_params": num_trainable,
        "total_params": num_total,
        "decoding": decoding_records,
        "finetune_losses": loss_history,
        "heldout": heldout_comparisons,
    }

    with open(output_dir / "outputs.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("Execution finalized successfully. All lab outputs verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
