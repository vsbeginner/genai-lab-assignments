"""Capstone Project: Parameter-Efficient Fine-Tuning and Evaluation of GPT-2.

Course: ETMMGA304 - Generative AI and Its Applications
Institution: K.R. Mangalam University (Master of Computer Applications)
Curriculum Reference: BTX 4 (Analyze), BTX 5 (Evaluate) & BTX 6 (Create)

This script trains a domain-adapted pedagogical study-tip generator using parameter-efficient
adaptation on GPT-2 (openai-community/gpt2, 124M parameters). It executes:
  1. Dataset ingestion (40 training pairs, 12 held-out evaluation topics) with mathematical
     zero-leakage assertion.
  2. Freezing layers 0-10 and optimizing strictly the final Transformer block (transformer.h[-1])
     with AdamW (lr=1e-4) over 10 epochs.
  3. Prefix prompt masking with Cross-Entropy loss (label=-100) to isolate loss to response tokens.
  4. Generation and logging of comparative outputs (Zero-Shot Base vs Adapted) on all held-out topics.

Outputs:
  - lab_outputs/study_tip_gpt2/: Fine-tuned model checkpoint and tokenizer configuration.
  - lab_outputs/train_log.json: Training loss trajectory across all 10 epochs.
  - lab_outputs/outputs.json: Paired predictions across all 12 held-out benchmark topics.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# -----------------------------------------------------------------------------
# Hyperparameters & Settings
# -----------------------------------------------------------------------------
MODEL_IDENTIFIER: str = "openai-community/gpt2"
RANDOM_SEED: int = 42
TOTAL_EPOCHS: int = 10
LEARNING_RATE: float = 1e-4  # Selected via training loss progression; held-out set kept strictly untouched
MAX_NEW_TOKENS: int = 45


def configure_environment(seed: int = RANDOM_SEED) -> Tuple[torch.device, Path]:
    """Configures deterministic seed environment and prepares output directory."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("lab_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return device, output_dir


def load_datasets(data_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Loads training pairs and held-out evaluation topics with strict leakage validation."""
    train_path = data_dir / "train.json"
    heldout_path = data_dir / "heldout.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_data = [tuple(item) for item in train_raw]

    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_data = json.load(f)

    # Formal validation of zero train/held-out overlap
    train_topics = {t.lower() for t, _ in train_data}
    heldout_topics = {h.lower() for h in heldout_data}
    overlap = train_topics & heldout_topics
    assert not overlap, f"Data leakage detected! Overlapping topics: {overlap}"

    return train_data, heldout_data


def train_adapter(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data: List[Tuple[str, str]],
    epochs: int,
    lr: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Executes parameter-efficient fine-tuning on the final Transformer block."""
    model.requires_grad_(False)
    model.transformer.h[-1].requires_grad_(True)

    optimizer = torch.optim.AdamW(model.transformer.h[-1].parameters(), lr=lr)
    loss_history: List[Dict[str, Any]] = []

    print(f"      Initiating fine-tuning over {epochs} epochs ({len(train_data)} examples per epoch)...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_samples = list(train_data)
        random.shuffle(epoch_samples)
        batch_losses: List[float] = []

        for topic, response in epoch_samples:
            prefix_tokens = tokenizer.encode(f"Topic: {topic}\nStudy tip:")
            target_tokens = tokenizer.encode(" " + response) + [tokenizer.eos_token_id]

            input_ids = torch.tensor([prefix_tokens + target_tokens], device=device)
            # Mask prompt tokens with -100 so loss evaluates only on response tokens
            labels = torch.tensor([[-100] * len(prefix_tokens) + target_tokens], device=device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        mean_loss = round(sum(batch_losses) / len(batch_losses), 3)
        loss_history.append({"epoch": epoch, "mean_loss": mean_loss})
        print(f"      Epoch [{epoch:02d}/{epochs:02d}] -> Mean Loss: {mean_loss:.3f}")

    model.eval()
    return loss_history


def generate_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    topic: str,
    device: torch.device,
    max_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Performs deterministic greedy text generation for a given topic prompt."""
    prompt = f"Topic: {topic}\nStudy tip:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        token_outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs.input_ids.size(1)
    new_tokens = token_outputs[0, prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    device, output_dir = configure_environment(RANDOM_SEED)
    data_dir = Path(__file__).resolve().parent / "data"

    print("=" * 70)
    print(f"Capstone Project: Domain-Adapted Study Tip Generator | Device: {device}")
    print("=" * 70)

    # 1. Dataset Ingestion & Verification
    print("[1/4] Ingesting & validating dataset splits...")
    train_data, heldout_topics = load_datasets(data_dir)
    print(f"      Training Corpus: {len(train_data)} instructional pairs")
    print(f"      Held-Out Evaluation Set: {len(heldout_topics)} unseen topics")

    # 2. Model Initialization
    print("\n[2/4] Loading pre-trained GPT-2 models (Base & Target Adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDENTIFIER).to(device).eval()
    tuned_model = AutoModelForCausalLM.from_pretrained(MODEL_IDENTIFIER).to(device)

    # 3. Parameter-Efficient Fine-Tuning
    print("\n[3/4] Commencing PEFT optimization loop (10 Epochs)...")
    training_log = train_adapter(tuned_model, tokenizer, train_data, TOTAL_EPOCHS, LEARNING_RATE, device)

    # Export fine-tuned model checkpoint
    export_dir = output_dir / "study_tip_gpt2"
    tuned_model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"\n      Saved adapted model checkpoint to: {export_dir}")

    # 4. Comparative Evaluation on Held-Out Topics
    print("\n[4/4] Generating comparative evaluation on held-out topics...")
    comparative_results: List[Dict[str, str]] = []

    for topic in heldout_topics:
        base_tip = generate_prediction(base_model, tokenizer, topic, device)
        tuned_tip = generate_prediction(tuned_model, tokenizer, topic, device)
        print(f"\n      Topic: '{topic}'")
        print(f"        Base Model    : {base_tip}")
        print(f"        Adapted Model : {tuned_tip}")
        comparative_results.append({
            "topic": topic,
            "base": base_tip,
            "adapted": tuned_tip,
        })

    # Serialize Artifacts
    train_metadata = {
        "seed": RANDOM_SEED,
        "epochs": TOTAL_EPOCHS,
        "lr": LEARNING_RATE,
        "train_size": len(train_data),
        "log": training_log,
    }

    with open(output_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)

    with open(output_dir / "outputs.json", "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Execution finalized successfully. Model and telemetry persisted.")
    print("=" * 70)


if __name__ == "__main__":
    main()
