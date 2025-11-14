#!/usr/bin/env python3
"""
Baseline GLUE fine-tuning with your own pre-trained DeBERTa-v3.
- No SenseMix, no WordNet.
- Works with any HF checkpoint (e.g., your ELECTRA-style DeBERTa-v3 discriminator).
- Logs to CSV + PNG curves.

Outputs (inside --output_dir):
- train_log.txt (all prints)
- train_log.csv (epoch, step, gstep, loss, metric, lr)
- learning_curve_loss.png
- learning_curve_metric.png
- best checkpoint: pytorch_model.bin + config + tokenizer
"""
import argparse, os, csv, random, math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW
import matplotlib.pyplot as plt

GLUE_KEYS = {
    "sst2": ("sentence", None, 2),
    "cola": ("sentence", None, 2),
    "mrpc": ("sentence1", "sentence2", 2),
    "qqp": ("question1", "question2", 2),
    "stsb": ("sentence1", "sentence2", 1),  # regression
    "mnli": ("premise", "hypothesis", 3),
    "qnli": ("question", "sentence", 2),
    "rte":  ("sentence1", "sentence2", 2),
}

def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_datasets(task_name: str, tokenizer, max_length: int):
    key1, key2, _ = GLUE_KEYS[task_name]
    ds = load_dataset("glue", task_name)

    def preprocess(ex):
        if key2 is None:
            enc = tokenizer(ex[key1], truncation=True, max_length=max_length)
        else:
            enc = tokenizer(ex[key1], ex[key2], truncation=True, max_length=max_length)
        enc["labels"] = ex.get("label", -1)
        return enc

    cols = ds["train"].column_names
    return ds.map(preprocess, batched=False, remove_columns=cols)

def build_model(model_name: str, num_labels: int, problem_type: str):
    # Load base config and create a classification head on top
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    if problem_type == "regression":
        cfg.problem_type = "regression"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
    return model

def compute_glue_metrics(task_name, preds, labels):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        mean_squared_error,
    )
    import scipy.stats as stats
    import numpy as np

    preds = np.array(preds)
    labels = np.array(labels)

    if task_name == "cola":
        # Matthews correlation
        return {"matthews": matthews_corrcoef(labels, preds)}

    elif task_name in ["mrpc", "qqp"]:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "mean": (acc + f1) / 2}

    elif task_name == "stsb":
        pearson = stats.pearsonr(preds, labels)[0]
        spearman = stats.spearmanr(preds, labels)[0]
        mse = mean_squared_error(labels, preds)
        return {"pearson": pearson, "spearman": spearman, "mse": mse}

    elif task_name in ["sst2", "qnli", "rte"]:
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    elif task_name == "mnli":
        # MNLI matched
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    else:
        raise ValueError(f"Unsupported task: {task_name}")


def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    txt_log = os.path.join(args.output_dir, "train_log.txt")
    csv_log = os.path.join(args.output_dir, "train_log.csv")
    log_f = open(txt_log, "w")

    def log_print(msg):
        print(msg)
        log_f.write(msg + "\n"); log_f.flush()

    # CSV header
    with open(csv_log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "global_step", "loss", "metric(acc_or_mse)", "lr"])

    tok_name = args.tokenizer_name if args.tokenizer_name else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=False)

    _, _, num_labels = GLUE_KEYS[args.task_name]
    problem_type = "regression" if num_labels == 1 else "single_label_classification"
    model = build_model(args.model_name, num_labels, problem_type).to(device)

    ds = make_datasets(args.task_name, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer)

    def loader(split, shuffle=False):
        name = "validation_matched" if (args.task_name=="mnli" and split=="validation") else split
        return torch.utils.data.DataLoader(ds[name], batch_size=args.batch_size,
                                           shuffle=shuffle, collate_fn=collator)

    train_loader = loader("train", shuffle=True) if args.do_train else None
    eval_name = "validation"
    eval_loader = loader(eval_name) if args.do_eval else None

    # Optimizer / Scheduler (Table-11-like defaults)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.learning_rate, eps=1e-6, betas=(0.9, 0.999))

    if args.do_train:
        t_total = len(train_loader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = None

    # Curves
    steps_hist, loss_hist, metric_hist = [], [], []

    # Print device info
    log_print(f"[Device] Using: {device}")
    if device.type == "cuda":
        log_print(f"[CUDA] {torch.cuda.get_device_name(0)} | {torch.cuda.memory_allocated(0)/1e9:.2f} GB used")

    best_metric = None
    best_path = os.path.join(args.output_dir, "best")
    os.makedirs(best_path, exist_ok=True)



    # ----------------
    # Training
    # ----------------
    global_step = 0
    if args.do_train:
        model.train()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_loader, start=1):
                batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                out = model(**batch)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

                # ----- batch metric -----
                with torch.no_grad():
                    logits = out.logits.detach()
                    if num_labels == 1:
                        y = batch["labels"].view(-1).float()
                        pred = logits.view(-1).float()
                        batch_metric = F.mse_loss(pred, y).item()
                    else:
                        y = batch["labels"].view(-1).long()
                        pred = logits.argmax(dim=-1).view(-1)
                        batch_metric = (pred == y).float().mean().item()

                global_step += 1
                steps_hist.append(global_step)
                loss_hist.append(loss.item())
                metric_hist.append(batch_metric)
                lr_now = optimizer.param_groups[0]["lr"]

                if global_step % args.logging_steps == 0:
                    metric_name = "acc" if num_labels > 1 else "mse"
                    log_print(
                        f"epoch {epoch+1} step {step} gstep {global_step} "
                        f"loss {loss.item():.4f} {metric_name} {batch_metric:.4f} lr {lr_now:.6f}"
                    )

            # --------- Eval each epoch ---------
            if args.do_eval:
                model.eval()
                preds, labels = [], []

                with torch.no_grad():
                    for eb in eval_loader:
                        labels.extend(eb["labels"].numpy().tolist())
                        eb = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in eb.items()}
                        logits = model(**eb).logits.detach().cpu().numpy()

                        if num_labels == 1:
                            preds.extend(logits.reshape(-1).tolist())
                        else:
                            preds.extend(logits.argmax(axis=-1).tolist())

                results = compute_glue_metrics(args.task_name, preds, labels)
                log_print(f"[EVAL epoch {epoch+1}] {results}")

                # best 기준 metric 선택
                if args.task_name == "stsb":
                    cur_metric = results["pearson"]
                elif args.task_name in ["mrpc", "qqp"]:
                    cur_metric = results["mean"]
                elif args.task_name == "cola":
                    cur_metric = results["matthews"]
                else:
                    cur_metric = results["accuracy"]

                improve = (best_metric is None) or (cur_metric > best_metric)
                if improve:
                    best_metric = cur_metric
                    model.save_pretrained(best_path)
                    tokenizer.save_pretrained(best_path)
                    log_print(f"[SAVE] New best saved to {best_path} (metric={cur_metric:.4f})")

                model.train()

        # --------- Final save (last) ---------
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        log_print("[INFO] Training done. Last model saved.")


    # ----------------
    # Final Evaluation (optional, uses final weights)
    # ----------------
    if args.do_eval:
        model.eval()
        from sklearn.metrics import accuracy_score
        import scipy.stats as stats
        preds, labels = [], []
        with torch.no_grad():
            for eb in eval_loader:
                labels.extend(eb["labels"].numpy().tolist())
                eb = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in eb.items()}
                el = model(**eb).logits.detach().cpu().numpy()
                if num_labels == 1:
                    preds.extend(el.reshape(-1).tolist())
                else:
                    preds.extend(el.argmax(axis=-1).tolist())
        if num_labels == 1:
            pearson = stats.pearsonr(np.array(preds), np.array(labels))[0]
            log_print(f"[EVAL final] Pearson: {pearson:.4f}")
        else:
            acc = (np.array(preds) == np.array(labels)).mean()
            log_print(f"[EVAL final] Accuracy: {acc:.4f}")

    # ----------------
    # Curves
    # ----------------
    try:
        plt.figure()
        plt.plot(steps_hist, loss_hist)
        plt.xlabel("Global Step"); plt.ylabel("Loss")
        plt.title("Training Loss vs Step")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "learning_curve_loss.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(steps_hist, metric_hist)
        plt.xlabel("Global Step"); plt.ylabel("Accuracy" if problem_type!="regression" else "MSE")
        ttl = "Training " + ("Accuracy" if problem_type!="regression" else "MSE") + " vs Step"
        plt.title(ttl)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "learning_curve_metric.png"), dpi=300)
        plt.close()
    except Exception as e:
        log_print(f"[WARN] Failed to save curves: {e}")

    log_f.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, required=True, choices=list(GLUE_KEYS.keys()))
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-xsmall",
                   help="Path or HF id of your pre-trained checkpoint (folder with config + weights).")
    p.add_argument("--tokenizer_name", type=str, default=None,
                   help="If None, use model_name tokenizer.")
    p.add_argument("--output_dir", type=str, default="runs-10p-baseline",)
    p.add_argument("--max_length", type=int, default=256)

    # Hyperparams (DeBERTaV3 Table 11 style)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)   # {1.5e-5, 2e-5, 2.5e-5, 3e-5}
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=1000)      # {50,100,500,1000}
    p.add_argument("--logging_steps", type=int, default=100)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)
