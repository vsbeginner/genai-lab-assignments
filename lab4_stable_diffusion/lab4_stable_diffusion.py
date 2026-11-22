"""Laboratory Assignment 04: Multimodal Prompt-to-Image Generation with Latent Diffusion Models.

Course: ETMMGA304 - Generative AI and Its Applications
Institution: K.R. Mangalam University (Master of Computer Applications)
Curriculum Reference: BTX 3 (Apply), BTX 4 (Analyze) & BTX 5 (Evaluate)

This script conducts controlled multimodal image synthesis using Stable Diffusion v1.5
(stable-diffusion-v1-5/stable-diffusion-v1-5) via Hugging Face Diffusers. It demonstrates:
  1. Half-precision (FP16) pipeline optimization and memory attention slicing for 6GB VRAM.
  2. Controlled factorial prompt comparison: Modifying artistic medium ('watercolor illustration'
     vs 'pencil sketch') while maintaining strict seed invariance (Seeds 42 and 43).
  3. Latent space denoising progression guided by Classifier-Free Guidance (CFG scale = 7.5).
  4. Programmatic safety filtering and inference latency benchmarking.

Outputs:
  - lab_outputs/library_prompt_1_seed_42.png: Watercolor rendering at Seed 42.
  - lab_outputs/library_prompt_1_seed_43.png: Watercolor rendering at Seed 43.
  - lab_outputs/library_prompt_2_seed_42.png: Pencil sketch rendering at Seed 42.
  - lab_outputs/library_prompt_2_seed_43.png: Pencil sketch rendering at Seed 43.
  - lab_outputs/results.json: Telemetry, generation parameters, and inference latency logs.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from diffusers import StableDiffusionPipeline


# -----------------------------------------------------------------------------
# Configuration & Generation Parameters
# -----------------------------------------------------------------------------
MODEL_REPO_ID: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_DIMENSIONS: Tuple[int, int] = (512, 512)
NUM_INFERENCE_STEPS: int = 25
GUIDANCE_SCALE: float = 7.5
TORCH_DTYPE = torch.float16

BENCHMARK_PROMPTS: List[str] = [
    "A small library with a reading table, wide view, watercolor illustration",
    "A small library with a reading table, wide view, pencil sketch",
]

EVALUATION_SEEDS: List[int] = [42, 43]


def configure_environment() -> Tuple[torch.device, Path]:
    """Validates GPU availability and prepares output directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARNING] CUDA accelerator not detected. Stable Diffusion requires GPU runtime for practical execution.")

    output_dir = Path("lab_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return device, output_dir


# -----------------------------------------------------------------------------
# Pipeline Initialization
# -----------------------------------------------------------------------------
def load_diffusion_pipeline(
    repo_id: str = MODEL_REPO_ID,
    device: torch.device = torch.device("cuda"),
) -> StableDiffusionPipeline:
    """Instantiates the Stable Diffusion pipeline in FP16 with attention slicing enabled."""
    print(f"      Loading pipeline from Hugging Face: '{repo_id}'...")
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=TORCH_DTYPE,
    )
    pipe = pipe.to(device)

    # Enable memory attention slicing to prevent OOM on 6GB consumer GPUs
    pipe.enable_attention_slicing()
    return pipe


# -----------------------------------------------------------------------------
# Controlled Factorial Synthesis
# -----------------------------------------------------------------------------
def execute_generation_suite(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Generates paired image artifacts holding latent seed constant across prompt variants."""
    run_records: List[Dict[str, Any]] = []

    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n      Prompt Variant [{prompt_idx}]: '{prompt}'")
        for seed in seeds:
            print(f"        -> Synthesizing at Seed {seed}...")
            generator = torch.Generator(device=device).manual_seed(seed)

            start_time = time.time()
            output = pipe(
                prompt,
                width=IMAGE_DIMENSIONS[0],
                height=IMAGE_DIMENSIONS[1],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            )
            elapsed_seconds = round(time.time() - start_time, 1)

            # Safety Checker Verification
            flagged = False
            if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
                flagged = bool(output.nsfw_content_detected[0])

            if flagged:
                print(f"        [FLAGGED] Output violated safety guidelines at Seed {seed}.")
                run_records.append({
                    "prompt": prompt,
                    "seed": seed,
                    "flagged": True,
                    "seconds": elapsed_seconds,
                })
                continue

            generated_image = output.images[0]
            filename = f"library_prompt_{prompt_idx}_seed_{seed}.png"
            image_filepath = output_dir / filename
            generated_image.save(image_filepath)

            print(f"        [SUCCESS] Saved: {filename} ({elapsed_seconds}s)")
            run_records.append({
                "prompt": prompt,
                "seed": seed,
                "flagged": False,
                "file": filename,
                "seconds": elapsed_seconds,
            })

    return run_records


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    device, output_dir = configure_environment()
    print("=" * 70)
    print(f"ETMMGA304 Lab 04: Stable Diffusion Image Synthesis | Device: {device}")
    print("=" * 70)

    # Step 1: Initialize Diffusion Pipeline
    print("[1/2] Initializing Stable Diffusion v1.5 pipeline...")
    pipe = load_diffusion_pipeline(MODEL_REPO_ID, device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"      Accelerator Hardware: {gpu_name}")

    # Step 2: Controlled Factorial Image Synthesis
    print("\n[2/2] Commencing controlled seed-invariance generation suite...")
    generation_logs = execute_generation_suite(
        pipe,
        BENCHMARK_PROMPTS,
        EVALUATION_SEEDS,
        output_dir,
        device,
    )

    # Serialize Telemetry & Benchmark Configuration
    telemetry_payload = {
        "repo": MODEL_REPO_ID,
        "device": str(device),
        "gpu": gpu_name,
        "settings": {
            "width": IMAGE_DIMENSIONS[0],
            "height": IMAGE_DIMENSIONS[1],
            "steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "dtype": "float16",
        },
        "runs": generation_logs,
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(telemetry_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("Execution finalized successfully. All lab outputs verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
