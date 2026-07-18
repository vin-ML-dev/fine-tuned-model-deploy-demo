"""
Phase 2 - Steps 5-8: QLoRA fine-tuning of Qwen2.5-1.5B-Instruct
on the TechNova company dataset. Designed for Google Colab T4 (16 GB, fp16).

- 4-bit NF4 quantization (bitsandbytes) + LoRA adapters (PEFT)
- Config-driven (configs/training_config.yaml)
- Checkpointing, early stopping, TensorBoard tracking
- Saves the best LoRA adapter to outputs/qlora-technova/final_adapter

Usage:
    python scripts/train_qlora.py --config configs/training_config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/training_config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    assert torch.cuda.is_available(), "GPU not found. In Colab: Runtime > Change runtime type > T4 GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    model_name = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. 4-bit quantized base model (QLoRA)
    # ------------------------------------------------------------------
    qcfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 3. LoRA adapters (PEFT)
    # ------------------------------------------------------------------
    lcfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        target_modules=lcfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Dataset (chat 'messages' format -> SFTTrainer applies chat template)
    # ------------------------------------------------------------------
    dcfg = cfg["data"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(ROOT / dcfg["train_file"]),
            "validation": str(ROOT / dcfg["val_file"]),
        },
    )
    # keep only the 'messages' column
    dataset = dataset.remove_columns(
        [c for c in dataset["train"].column_names if c != "messages"]
    )
    print(dataset)

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    tcfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=str(ROOT / tcfg["output_dir"]),
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        save_strategy=tcfg["save_strategy"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        fp16=tcfg["fp16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        optim=tcfg["optim"],
        seed=tcfg["seed"],
        report_to=tcfg["report_to"],
        max_length=cfg["model"]["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping"]["patience"]
            )
        ],
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    train_result = trainer.train()
    metrics = train_result.metrics
    print(f"Training complete. Final train loss: {metrics.get('train_loss'):.4f}")

    # ------------------------------------------------------------------
    # 7. Save best adapter + tokenizer + run metadata (model registry step)
    # ------------------------------------------------------------------
    final_dir = ROOT / tcfg["output_dir"] / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    eval_metrics = trainer.evaluate()
    registry_entry = {
        "base_model": model_name,
        "adapter_path": str(final_dir),
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "lora_config": lcfg,
        "epochs_run": metrics.get("epoch"),
    }
    with open(ROOT / tcfg["output_dir"] / "model_card.json", "w") as f:
        json.dump(registry_entry, f, indent=2)

    print(f"Adapter saved to: {final_dir}")
    print(f"Eval loss: {eval_metrics.get('eval_loss'):.4f}")
    print("Run `tensorboard --logdir outputs/qlora-technova` to view training curves.")


if __name__ == "__main__":
    main()
