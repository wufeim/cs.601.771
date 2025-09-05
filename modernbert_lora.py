import argparse
import logging
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    set_seed
)

from utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification head.")

    parser.add_argument("--exp_name", type=str, default="modernbert_head")
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")

    parser.add_argument("--dataset_name", type=str, default="wics/strategy-qa")
    parser.add_argument("--dataset_revision", type=str, default="refs/convert/parquet")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size_per_gpu", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--total_train_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=int, default=0.06)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=500)

    parser.add_argument("--wandb_project", type=str, default="modernbert")

    args = parser.parse_args()
    args.eval_every = args.total_train_steps // 10
    args.checkpoint_every = args.total_train_steps // 10
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    return args


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=args.output_dir)

    if accelerator.is_main_process:
        setup_logging(args.output_dir, log_name="train")
        accelerator.init_trackers(
            args.wandb_project,
            config=vars(args),
            init_kwargs={
                "wandb": {"name": args.exp_name}
            })

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_dataloader, val_dataloader, test_dataloader = build_dataset(
        args, tokenizer, collator)

    model = build_model(args).to(accelerator.device)
    lora_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    num_epochs = math.ceil(args.total_train_steps * accelerator.num_processes / len(train_dataloader))
    warmup_steps = int(args.warmup_ratio * args.total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, args.total_train_steps * accelerator.num_processes)

    print_model_stats(model)

    model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler)

    global_step, best_val_acc = 1, 0.0
    for epoch in range(1, num_epochs + 1):
        for _, batch in enumerate(train_dataloader):
            model.train()
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(trainable_params, args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % args.log_every == 0:
                pred = outputs.logits.argmax(-1)
                pred_g = accelerator.gather_for_metrics(pred)
                label_g = accelerator.gather_for_metrics(batch["labels"])
                correct = (pred_g == label_g).sum().item()
                total = label_g.numel()
                accuracy = correct / total
                if accelerator.is_main_process:
                    logging.info(
                        f"[Step {global_step}] train_loss: {loss.item():.4f} "
                        f"| train_acc: {accuracy:.4f}"
                        f"| lr: {scheduler.get_last_lr()[0]:.4f}")
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/acc": accuracy,
                        "train/lr": scheduler.get_last_lr()[0]
                    }, step=global_step+1)

            checkpoint_best_val = False
            if global_step % args.eval_every == 0:
                model.eval()
                all_pred, all_lbl = [], []
                with torch.inference_mode():
                    for batch in val_dataloader:
                        outputs = model(**batch)
                        pred = outputs.logits.argmax(-1)
                        pred_g = accelerator.gather_for_metrics(pred)
                        label = batch["labels"]
                        label_g = accelerator.gather_for_metrics(label)
                        all_pred.append(pred_g.cpu())
                        all_lbl.append(label_g.cpu())
                all_pred = torch.cat(all_pred)
                all_lbl = torch.cat(all_lbl)
                val_acc = (all_pred == all_lbl).float().mean().item()

                if accelerator.is_main_process:
                    logging.info(
                        f"[Step {global_step}] val_acc: {val_acc:.4f}")
                    accelerator.log({
                        "val/acc": val_acc}, step=global_step+1)

                if val_acc >= best_val_acc:
                    checkpoint_best_val = True
                    best_val_acc = val_acc

            if checkpoint_best_val or global_step % args.checkpoint_every == 0:
                save_dir = os.path.join(args.output_dir, "ckpts", f"step-{global_step:06d}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(save_dir)

                if checkpoint_best_val:
                    best_dir = os.path.join(args.output_dir, "ckpts", "best")
                    if os.path.islink(best_dir):
                        os.remove(best_dir)
                    os.symlink(
                        os.path.abspath(save_dir), os.path.abspath(best_dir))

            global_step += 1

            if global_step >= args.total_train_steps:
                break

        if global_step >= args.total_train_steps:
            break

    logging.info(f'Training completed. Best val accuracy: {best_val_acc:.4f}')

    # Final testing
    best_dir = os.path.join(args.output_dir, "ckpts", "best")
    model = AutoModelForMaskedLM.from_pretrained(
        best_dir).to(accelerator.device)
    model.eval()
    all_pred, all_lbl = [], []
    with torch.inference_mode():
        for batch in test_dataloader:
            outputs = model(**batch)
            pred = outputs.logits.argmax(-1)
            pred_g = accelerator.gather_for_metrics(pred)
            label = batch["labels"]
            label_g = accelerator.gather_for_metrics(label)
            all_pred.append(pred_g.cpu())
            all_lbl.append(label_g.cpu())
    all_pred = torch.cat(all_pred)
    all_lbl = torch.cat(all_lbl)
    test_acc = (all_pred == all_lbl).float().mean().item()
    logging.info(f'Final test performance: {test_acc:.4f}')
    accelerator.log({"test/acc": test_acc}, step=global_step)


def build_dataset(args, tokenizer, collator):
    dataset = load_dataset(
        args.dataset_name, revision=args.dataset_revision)["test"]

    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    val_test_dataset = splits["test"]

    splits = val_test_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = splits["train"]
    test_dataset = splits["test"]

    def preprocess(batch):
        text = batch["question"]
        result = tokenizer(text, truncation=True, max_length=args.max_length)
        L = len(result["input_ids"])
        result = tokenizer(text, truncation=True, max_length=args.max_length, padding="max_length")
        result["labels"] = [-100 for _ in range(len(result["input_ids"]))]
        result["labels"][L-1] = 5088 if batch["answer"] else 5653
        return result

    remove_columns = [c for c in train_dataset.column_names]
    train_dataset = train_dataset.map(preprocess, remove_columns=remove_columns)
    val_dataset = val_dataset.map(preprocess, remove_columns=remove_columns)
    test_dataset = test_dataset.map(preprocess, remove_columns=remove_columns)

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collator)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator)

    logging.info(f"Number of training batches: {len(train_dataloader)}")
    logging.info(f"Number of validation batches: {len(val_dataloader)}")
    logging.info(f"Number of test batches: {len(test_dataloader)}")

    return train_dataloader, val_dataloader, test_dataloader


def build_model(args):
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    return model


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")


if __name__ == "__main__":
    main()
