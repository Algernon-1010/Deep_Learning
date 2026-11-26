"""Course lab: scratch regression, activations, Fashion-MNIST and MNIST.

Run --help for options. No experiment result is hard-coded.
The scratch network uses ParameterList for registration only; its affine
operations, ReLU and SGD updates are written explicitly.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


FASHION_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_figure(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def regression(out):
    out.mkdir(parents=True, exist_ok=True)
    seed_all(42)
    true_w = torch.tensor([[3.0], [2.0]])
    true_b = 5.0
    x = torch.randn(1000, 2)
    y = x @ true_w + true_b + 0.01 * torch.randn(1000, 1)
    xt = torch.randn(200, 2)
    yt = xt @ true_w + true_b + 0.01 * torch.randn(200, 1)
    w = (0.01 * torch.randn(2, 1)).requires_grad_()
    b = torch.zeros(1, requires_grad=True)
    with torch.no_grad():
        rows = [{"epoch": 0, "train_mse": ((x @ w + b-y)**2).mean().item(),
                 "w1": w[0].item(), "w2": w[1].item(), "b": b.item()}]
    for epoch in range(1, 11):
        for ix in torch.randperm(len(x)).split(10):
            loss = ((x[ix] @ w + b - y[ix]) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                w -= 0.1 * w.grad
                b -= 0.1 * b.grad
                w.grad = None
                b.grad = None
        with torch.no_grad():
            rows.append({"epoch": epoch, "train_mse": ((x @ w + b-y)**2).mean().item(),
                         "w1": w[0].item(), "w2": w[1].item(), "b": b.item()})
        print("Regression", rows[-1], flush=True)
    with torch.no_grad():
        pred = xt @ w + b
        metrics = {"true_w": [3.0, 2.0], "true_b": true_b,
                   "learned_w": w.flatten().tolist(), "learned_b": b.item(),
                   "weight_l2_error": (w-true_w).norm().item(),
                   "bias_abs_error": abs(b.item()-true_b),
                   "train_mse": rows[-1]["train_mse"],
                   "heldout_mse": ((pred-yt)**2).mean().item(),
                   "train_n": 1000, "heldout_n": 200, "seed": 42,
                   "epochs": 10, "lr": 0.1, "batch_size": 10}
    dump_json(out / "metrics.json", metrics)
    write_csv(out / "history.csv", rows)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].semilogy([r["epoch"] for r in rows], [r["train_mse"] for r in rows], "o-")
    axs[0].set(xlabel="Epoch", ylabel="Training MSE")
    # Remove x2's known contribution; raw y versus x1 has confounding variation.
    axs[1].scatter(x[:, 0], (y - 2*x[:, 1:2]).flatten(), s=4, alpha=.4)
    grid = torch.linspace(-3, 3, 100)
    axs[1].plot(grid, 3*grid+5, "r-", label="3*x1 + 5")
    axs[1].set(xlabel="x1", ylabel="y - 2*x2")
    axs[1].legend()
    axs[2].scatter(yt.flatten(), pred.flatten(), s=8, alpha=.6)
    lo, hi = yt.min().item(), yt.max().item()
    axs[2].plot([lo, hi], [lo, hi], "r--")
    axs[2].set(xlabel="Held-out target", ylabel="Held-out prediction")
    fig.tight_layout()
    save_figure(fig, out / "regression.png")


def activations(out):
    out.mkdir(parents=True, exist_ok=True)
    funcs = {"ReLU": torch.relu, "Sigmoid": torch.sigmoid, "Tanh": torch.tanh}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    rows = []
    for j, (name, fn) in enumerate(funcs.items()):
        x = torch.linspace(-8, 8, 1001, requires_grad=True)
        y = fn(x)
        grad, = torch.autograd.grad(y.sum(), x)
        if name == "ReLU":
            analytic = (x > 0).float()
        elif name == "Sigmoid":
            analytic = y * (1-y)
        else:
            analytic = 1-y*y
        err = (grad-analytic).abs().max().item()
        if err > 1e-6:
            raise AssertionError(f"Derivative mismatch: {name} {err}")
        rows.append({"activation": name, "max_derivative_abs_error": err,
                     "relu_zero_convention": "PyTorch selects 0 at x=0" if j == 0 else ""})
        xx = x.detach().numpy()
        axes[0, j].plot(xx, y.detach().numpy())
        axes[0, j].set(title=name, xlabel="x", ylabel="f(x)")
        axes[1, j].plot(xx, grad.numpy(), label="autograd")
        axes[1, j].plot(xx, analytic.detach().numpy(), "--", label="analytic")
        axes[1, j].set(xlabel="x", ylabel="Derivative")
        axes[1, j].legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.tight_layout()
    save_figure(fig, out / "activations.png")
    write_csv(out / "derivative_check.csv", rows)
    print("Activation derivative checks passed.", flush=True)


class ScratchMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        dims = [784, *hidden, 10]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(a, b)*0.01) for a, b in zip(dims[:-1], dims[1:])])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(b)) for b in dims[1:]])

    def forward(self, x):
        h = x.reshape(x.shape[0], 784)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights)-1:
                # Explicit ReLU; chooses zero gradient at zero.
                h = torch.where(h > 0, h, torch.zeros_like(h))
        return h  # logits, with NO softmax before cross_entropy


def scratch_sgd(parameters, lr):
    with torch.no_grad():
        for p in parameters:
            if p.grad is not None:
                p -= lr*p.grad
                p.grad = None


@torch.inference_mode()
def evaluate(model, loader, device, collect=False):
    model.eval()
    total_loss = correct = count = 0
    ys, ps, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        count += len(y)
        if collect:
            ys.append(y.cpu())
            ps.append(pred.cpu())
            probs.append(logits.softmax(1).cpu())
    result = {"loss": total_loss/count, "accuracy": correct/count}
    if collect:
        result.update(y=torch.cat(ys), pred=torch.cat(ps), probs=torch.cat(probs))
    return result


def md5_matches(path, expected):
    if not path.is_file():
        return False
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def prepare_fashion_data(data_root):
    """Recover interrupted downloads using official sources and MD5 checks.

    Never append an unverified partial file. Keep final filenames unchanged
    until a complete download has passed the dataset publisher's checksum.
    """
    raw = data_root / "FashionMNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        "https://fashion-mnist.s3.eu-central-1.amazonaws.com/",
    ]
    for filename, expected_md5 in torchvision.datasets.FashionMNIST.resources:
        archive = raw / filename
        if not md5_matches(archive, expected_md5):
            part = raw / (filename + f".part.{os.getpid()}")
            errors = []
            downloaded = False
            for base in mirrors:
                for attempt in range(1, 3):
                    url = base + filename
                    try:
                        print(f"Downloading {filename} (attempt {attempt}/2) from {base}", flush=True)
                        request = Request(url, headers={"User-Agent": "mlp-course-lab/1.1"})
                        with urlopen(request, timeout=30) as response, part.open("wb") as f:
                            total = int(response.headers.get("Content-Length", 0))
                            received, last_print = 0, time.monotonic()
                            while True:
                                block = response.read(1024*1024)
                                if not block:
                                    break
                                f.write(block)
                                received += len(block)
                                if time.monotonic()-last_print >= 3:
                                    suffix = f" / {total/1e6:.1f} MB" if total else " MB"
                                    print(f"  {received/1e6:.1f}{suffix}", flush=True)
                                    last_print = time.monotonic()
                        if not md5_matches(part, expected_md5):
                            raise RuntimeError("download incomplete or MD5 checksum mismatch")
                        part.replace(archive)
                        downloaded = True
                        print(f"  MD5 verified: {filename}", flush=True)
                        break
                    except (OSError, ValueError, RuntimeError, HTTPException) as exc:
                        errors.append(f"{url}: {type(exc).__name__}: {exc}")
                        print(f"  Download failed; retrying or trying the next official source: {exc}", flush=True)
                if downloaded:
                    break
            if not downloaded:
                raise RuntimeError("Fashion-MNIST download failed. Existing verified files were preserved.\n"
                                   + "\n".join(errors))
        extracted = raw / filename.removesuffix(".gz")
        temporary = raw / (extracted.name + f".extracting.{os.getpid()}")
        # Always derive IDX files from the verified archive, recovering any
        # truncated extraction left by a previous interrupted process.
        with gzip.open(archive, "rb") as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        temporary.replace(extracted)


def load_dataset(kind, data_root):
    cls = torchvision.datasets.FashionMNIST if kind == "fashion" else torchvision.datasets.MNIST
    if kind == "fashion":
        prepare_fashion_data(data_root)
    train = cls(str(data_root), train=True, download=kind != "fashion")
    test = cls(str(data_root), train=False, download=kind != "fashion")
    # Equivalent to ToTensor for uint8 grayscale images, but vectorized once.
    train_ds = TensorDataset(train.data.unsqueeze(1).float()/255.0, train.targets.long())
    test_ds = TensorDataset(test.data.unsqueeze(1).float()/255.0, test.targets.long())
    order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(2026))
    n_val = 6000
    names = FASHION_NAMES if kind == "fashion" else [str(i) for i in range(10)]
    return Subset(train_ds, order[n_val:].tolist()), Subset(train_ds, order[:n_val].tolist()), test_ds, names, order


def diagnostic_plots(out, test_ds, names, result):
    y, pred, prob = result["y"], result["pred"], result["probs"]
    cm = torch.bincount(y*10+pred, minlength=100).reshape(10, 10).numpy()
    np.savetxt(out / "confusion_counts.csv", cm, delimiter=",", fmt="%d")
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(10), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(10), where=cm.sum(1) != 0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(10), where=precision+recall != 0)
    write_csv(out / "class_metrics.csv", [dict(label=i, name=names[i], precision=precision[i],
              recall=recall[i], f1=f1[i], support=int(cm[i].sum())) for i in range(10)])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > .55 else "black")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True", title="Confusion matrix: counts; color = row proportion")
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    save_figure(fig, out / "confusion.png")

    wrong = (y != pred).nonzero().flatten().tolist()
    for fname, ids in [("predictions.png", list(range(16))), ("errors.png", wrong[:16])]:
        if not ids:
            continue
        fig, axs = plt.subplots(4, 4, figsize=(11, 10))
        for ax in axs.flat:
            ax.axis("off")
        for ax, ix in zip(axs.flat, ids):
            ax.imshow(test_ds[ix][0].squeeze(), cmap="gray")
            p = pred[ix].item()
            ax.set_title(f"#{ix} True: {names[y[ix].item()]}\nPred: {names[p]} ({prob[ix,p]:.2f})",
                         fontsize=8, color="darkgreen" if y[ix] == pred[ix] else "darkred")
        fig.tight_layout()
        save_figure(fig, out / fname)
    # Fixed index, not cherry-picked for a successful prediction.
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].imshow(test_ds[0][0].squeeze(), cmap="gray")
    axs[0].set_title(f"Index 0; true: {names[y[0].item()]}")
    axs[0].axis("off")
    axs[1].barh(names, prob[0].numpy())
    axs[1].set(xlim=(0, 1), xlabel="Softmax score", title=f"Predicted: {names[pred[0].item()]}")
    fig.tight_layout()
    save_figure(fig, out / "probability.png")
    write_csv(out / "predictions.csv", [dict(index=i, true=y[i].item(), predicted=pred[i].item(),
              score=prob[i, pred[i]].item()) for i in range(len(y))])
    return float(f1.mean()), len(wrong)


def train_one(kind, config, hidden, seed, datasets, args, root):
    train_ds, val_ds, test_ds, names, _ = datasets
    out = root / kind / f"{config}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    device = torch.device(args.device)
    model = ScratchMLP(hidden).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    train_eval = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
    history, best_state = [], None
    best_val, best_epoch = math.inf, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect learning rate and inputs.")
            loss.backward()
            scratch_sgd(model.parameters(), args.lr)
        tr = evaluate(model, train_eval, device)
        va = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch, train_loss=tr["loss"], val_loss=va["loss"],
                            train_accuracy=tr["accuracy"], val_accuracy=va["accuracy"]))
        if va["loss"] < best_val:
            best_val, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{kind} {config} seed={seed} epoch={epoch}/{args.epochs} "
              f"train_loss={tr['loss']:.4f} train_acc={tr['accuracy']:.4%} "
              f"val_loss={va['loss']:.4f} val_acc={va['accuracy']:.4%}", flush=True)
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(best_state)
    tr = evaluate(model, train_eval, device)
    va = evaluate(model, val_loader, device)
    te = evaluate(model, test_loader, device, collect=True)  # one final test evaluation
    macro_f1, wrong_n = diagnostic_plots(out, test_ds, names, te)
    metrics = dict(dataset=kind, config=config, hidden=list(hidden), seed=seed,
                   parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                   train_accuracy=tr["accuracy"], val_accuracy=va["accuracy"],
                   test_accuracy=te["accuracy"], test_loss=te["loss"], macro_f1=macro_f1,
                   error_count=wrong_n, generalization_gap=tr["accuracy"]-te["accuracy"],
                   fit_wall_seconds=fit_seconds, epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, initialization="normal std 0.01; bias zero",
                   train_n=len(train_ds), val_n=len(val_ds), test_n=len(test_ds),
                   split_seed=2026, selection="minimum validation cross-entropy")
    dump_json(out / "metrics.json", metrics)
    write_csv(out / "history.csv", history)
    torch.save(dict(state_dict=best_state, config=metrics), out / "checkpoint.pt")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for prefix in ["train", "val"]:
        axs[0].plot([r["epoch"] for r in history], [r[prefix+"_loss"] for r in history], label=prefix)
        axs[1].plot([r["epoch"] for r in history], [100*r[prefix+"_accuracy"] for r in history], label=prefix)
    axs[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axs[1].set(xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axs:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=.5)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, out / "curves.png")
    print(f"FINAL {kind} {config} seed={seed}: test_acc={te['accuracy']:.4%}, macro_F1={macro_f1:.4f}", flush=True)
    return metrics


def summarize(root, rows):
    if not rows:
        return
    write_csv(root / "all_metrics.csv", rows)
    grouped = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["config"]), []).append(r)
    aggregated = []
    for (dataset, config), rs in grouped.items():
        item = dict(dataset=dataset, config=config, n_seeds=len(rs), parameters=rs[0]["parameters"])
        for field in ["val_accuracy", "test_accuracy", "macro_f1", "generalization_gap", "fit_wall_seconds"]:
            values = [r[field] for r in rs]
            item[field+"_mean"] = statistics.mean(values)
            item[field+"_sample_std"] = statistics.stdev(values) if len(values) > 1 else ""
        aggregated.append(item)
    write_csv(root / "aggregate_metrics.csv", aggregated)
    lines = ["# Actual measured results", "", "Accuracy and gaps below use percentage points. F1 uses 0-1.", "",
             "|Dataset|Model|Seeds|Parameters|Validation accuracy %|Test accuracy %|Macro F1|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregated:
        def fmt(field, percent=False):
            mult = 100 if percent else 1
            value = f"{r[field+'_mean']*mult:.4f}"
            sd = r[field+'_sample_std']
            return value + (f" +/- {sd*mult:.4f}" if sd != "" else " (single seed)")
        lines.append(f"|{r['dataset']}|{r['config']}|{r['n_seeds']}|{r['parameters']}|"
                     f"{fmt('val_accuracy', True)}|{fmt('test_accuracy', True)}|{fmt('macro_f1')}|")
    lines += ["", "Standard deviation is the sample standard deviation across training seeds; it is not a confidence interval.",
              "Each run uses the checkpoint with the lowest validation loss. Do not choose or tune models by test results."]
    (root / "measured_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["basics", "fashion", "mnist", "all"], default="all")
    p.add_argument("--suite", choices=["baseline", "compare"], default="baseline",
                   help="compare adds softmax, hidden=10 and hidden=256,128 on Fashion-MNIST")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "mlp_results")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        p.error("epochs, batch-size, threads and lr must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    root = args.out.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    dump_json(root / "environment.json", dict(python=sys.version, executable=sys.executable,
              platform=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__,
              numpy=np.__version__, matplotlib=matplotlib.__version__, device=args.device,
              arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    print(f"Output directory: {root}", flush=True)
    if args.task in ["basics", "all"]:
        regression(root / "01_regression")
        activations(root / "02_activations")
    rows = []
    for kind in ["fashion", "mnist"]:
        if args.task not in [kind, "all"]:
            continue
        datasets = load_dataset(kind, args.data.resolve())
        (root / kind).mkdir(exist_ok=True)
        order = datasets[-1].numpy()
        np.savez(root / kind / "split_indices.npz", train=order[6000:], validation=order[:6000])
        configs = {"mlp256": (256,)}
        if kind == "fashion" and args.suite == "compare":
            configs = {"softmax": (), "mlp10": (10,), "mlp256": (256,), "mlp256_128": (256,128)}
        for config, hidden in configs.items():
            for seed in args.seeds:
                rows.append(train_one(kind, config, hidden, seed, datasets, args, root))
                summarize(root, rows)
        del datasets
    print(f"DONE. Results: {root}", flush=True)


if __name__ == "__main__":
    main()
