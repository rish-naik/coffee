#!/usr/bin/env python
"""
Coffee leaf stress classifier (BRACOL): training, prediction, evaluation and Grad-CAM heatmaps.

Model: MobileNetV3-Large with two heads on a shared trunk
  * main head  - which stress predominates (healthy / miner / rust / phoma / cercospora), softmax
  * flags head - is each stressor present (miner / rust / phoma / cercospora), sigmoid

The main stress is the primary output. The flags are reported only as a low-confidence
"possible additional stress" with a raw probability.

Usage:
  python leaf_classifier.py train                         # train on all labeled leaves
  python leaf_classifier.py train --holdout 0.1           # keep 10% aside for a sanity check
  python leaf_classifier.py predict leaf.jpg              # a file or a folder
  python leaf_classifier.py predict leaf.jpg --heatmap-dir heatmaps
  python leaf_classifier.py evaluate --ids outputs/holdout_ids.csv
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAIN_NAMES = ["healthy", "miner", "rust", "phoma", "cercospora"]   # predominant_stress codes 0..4
FLAG_NAMES = ["miner", "rust", "phoma", "cercospora"]              # flag columns, same order as codes 1..4
IGNORE = -100                                                      # code 5 ("multiple") has no main label
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SECONDARY_DISPLAY_MIN = 0.30     # show a possible additional stress only if its probability is at least this
WEIGHT_DECAY = 1e-4
LR_HEAD, LR_FINETUNE = 1e-3, 1e-4


# --------------------------------------------------------------------------- utilities
def seed_everything(seed):
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    """Return the device used for training and inference (CPU only in this version)."""
    return torch.device("cpu")


def load_table(csv_path, image_dir):
    """Read the labels CSV and attach an image path to each row.

    Images are expected at `<image_dir>/<id>.jpg`; rows without one are dropped. Adds `path`, `code` (predominant_stress as int) and `main`
    (the main-head target: `code` for 0..4, IGNORE for code 5, "multiple").
    """
    df = pd.read_csv(csv_path)
    need = {"id", "predominant_stress", *FLAG_NAMES}
    if not need <= set(df.columns):
        raise ValueError(f"CSV is missing columns: {sorted(need - set(df.columns))}")
    if not Path(image_dir).is_dir():
        raise SystemExit(f"Image folder not found: {image_dir}  (pass --images with the folder that holds the leaf images)")
    df["path"] = [str(Path(image_dir) / f"{i}.jpg") for i in df["id"]]
    df.loc[~df["path"].map(os.path.exists), "path"] = np.nan
    print(f"CSV rows: {len(df)}, without an image file: {int(df['path'].isna().sum())} (dropped)")
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No CSV row matched an image in {image_dir}. Check --images and how the files are named.")
    df["code"] = df["predominant_stress"].astype(int)
    df["main"] = np.where(df["code"] <= 4, df["code"], IGNORE).astype(np.int64)
    print("Leaves per predominant stress:", df["code"].value_counts().sort_index().to_dict())
    return df


# --------------------------------------------------------------------------- data
class RandomDegrade:
    """Augmentation that simulates a low-quality capture: downsample, then upsample back."""

    def __init__(self, p=0.3, min_scale=0.25, max_scale=0.9):
        self.p, self.min_scale, self.max_scale = p, min_scale, max_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def train_transform():
    """Training pipeline: resize, random degradation, flips, rotation, colour jitter, normalize."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), RandomDegrade(p=0.3),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def eval_transform():
    """Deterministic pipeline for evaluation and prediction: resize and normalize."""
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    """Yields (image tensor, 4 flag targets, main-stress target) for each row of a table from `load_table`."""

    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.flags = (df[FLAG_NAMES].to_numpy() > 0).astype(np.float32)
        self.main = df["main"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.from_numpy(self.flags[i]), int(self.main[i])


def make_loader(df, transform, batch_size, workers, shuffle=False, pin_memory=False):
    """Build a DataLoader over the rows of `df`."""
    return DataLoader(LeafDataset(df, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory)


# --------------------------------------------------------------------------- model
class LeafNet(nn.Module):
    """MobileNetV3-Large trunk with a flags head (4 logits) and a main-stress head (5 logits)."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        self.features, self.avgpool = m.features, m.avgpool
        self.trunk = m.classifier[:-1]                       # Linear(960,1280) + Hardswish + Dropout
        hidden = m.classifier[-1].in_features
        self.head_flags = nn.Linear(hidden, len(FLAG_NAMES))
        self.head_main = nn.Linear(hidden, len(MAIN_NAMES))

    def forward(self, x):
        """Return (flag logits, main-stress logits)."""
        h = self.trunk(self.avgpool(self.features(x)).flatten(1))
        return self.head_flags(h), self.head_main(h)


def set_backbone_trainable(model, trainable):
    """Freeze or unfreeze the convolutional backbone (`model.features`)."""
    for p in model.features.parameters():
        p.requires_grad = trainable


def load_model(model_dir, device):
    """Load a trained model and its metadata from `model_dir`. Returns (model, meta)."""
    model_dir = Path(model_dir)
    if not (model_dir / "model.pt").exists():
        raise SystemExit(f"No model.pt in {model_dir} (pass --model-dir with the folder you trained into)")
    meta = json.loads((model_dir / "meta.json").read_text())
    model = LeafNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    return model, meta


# --------------------------------------------------------------------------- training
def run_epoch(model, loader, device, bce, ce, lam, optimizer=None, desc=""):
    """Run one pass over `loader`; trains when `optimizer` is given, otherwise only evaluates.

    The loss is the flags loss (`bce`) plus `lam` times the main-stress loss (`ce`).
    Returns (mean loss, main-stress accuracy, mean F1 over the four flags at a 0.5 threshold).
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    flag_pred, flag_true, main_pred, main_true = [], [], [], []
    with torch.set_grad_enabled(train):
        for x, yf, ym in tqdm(loader, desc=desc, leave=False):
            x, yf, ym = x.to(device), yf.to(device), ym.to(device)
            lf, lm = model(x)
            loss = bce(lf, yf)
            if lam > 0 and (ym != IGNORE).any():
                loss = loss + lam * ce(lm, ym)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            flag_pred.append((torch.sigmoid(lf) >= 0.5).int().cpu())
            flag_true.append(yf.int().cpu())
            main_pred.append(lm.argmax(1).cpu())
            main_true.append(ym.cpu())
    fp, fy = torch.cat(flag_pred).numpy(), torch.cat(flag_true).numpy()
    mp, my = torch.cat(main_pred).numpy(), torch.cat(main_true).numpy()
    keep = my != IGNORE
    flag_f1 = float(np.mean([f1_score(fy[:, k], fp[:, k], zero_division=0) for k in range(fy.shape[1])]))
    main_acc = float((mp[keep] == my[keep]).mean()) if keep.any() else float("nan")
    return total / n, main_acc, flag_f1


def train(args):
    """Train the model (head only, then full fine-tune) and save model.pt and meta.json to `args.out`.

    With --holdout, a stratified fraction of the leaves is kept out of training, its ids are written to
    holdout_ids.csv, and a main-stress report on those leaves is printed at the end. The final epoch is
    kept; there is no checkpoint selection.
    """
    seed_everything(args.seed)
    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_table(args.csv, args.images)
    if args.holdout > 0:
        tr_idx, ho_idx = train_test_split(np.arange(len(df)), test_size=args.holdout,
                                          stratify=df["code"], random_state=args.seed)
        tr_df, ho_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
        ho_df[["id"]].to_csv(out / "holdout_ids.csv", index=False)
        print(f"Training on {len(tr_df)} leaves, holding out {len(ho_df)} for a sanity check.")
    else:
        tr_df, ho_df = df, None
        print(f"Training on all {len(tr_df)} leaves (no holdout: rely on the cross-validation estimate).")

    # Loss weights come from the training rows only.
    flags = (tr_df[FLAG_NAMES].to_numpy() > 0).astype(np.float32)
    pos = flags.sum(0)
    pos_w = np.clip(np.sqrt((len(tr_df) - pos) / np.maximum(pos, 1)), 1.0, 4.0)      # soft weight for rarer stressors
    mc = np.bincount(tr_df["main"][tr_df["main"] != IGNORE], minlength=len(MAIN_NAMES))
    main_w = mc.sum() / (len(MAIN_NAMES) * np.maximum(mc, 1))                         # inverse-frequency class weights
    print("pos_weight (flags):", dict(zip(FLAG_NAMES, pos_w.round(2))))
    print("class weights (main):", dict(zip(MAIN_NAMES, main_w.round(2))))

    pin = device.type == "cuda"
    train_loader = make_loader(tr_df, train_transform(), args.batch_size, args.workers, shuffle=True, pin_memory=pin)
    ho_loader = (make_loader(ho_df, eval_transform(), args.batch_size, args.workers, pin_memory=pin)
                 if ho_df is not None else None)

    model = LeafNet(pretrained=True).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(main_w, dtype=torch.float32, device=device), ignore_index=IGNORE)

    def phase(epochs, lr, trainable, label):
        set_backbone_trainable(model, trainable)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        for ep in range(epochs):
            loss, acc, f1 = run_epoch(model, train_loader, device, bce, ce, args.lambda_main, opt, f"{label} {ep+1}/{epochs}")
            sched.step()
            line = f"[{label}] epoch {ep+1}/{epochs}  train loss {loss:.3f}  main acc {acc:.3f}  flag F1 {f1:.3f}"
            if ho_loader is not None:
                hl, ha, hf = run_epoch(model, ho_loader, device, bce, ce, args.lambda_main, None, "holdout")
                line += f"  | holdout loss {hl:.3f} main acc {ha:.3f} flag F1 {hf:.3f}"
            tqdm.write(line)

    phase(args.epochs_head, LR_HEAD, False, "head")
    phase(args.epochs_finetune, LR_FINETUNE, True, "finetune")

    torch.save(model.state_dict(), out / "model.pt")
    meta = {"main_names": MAIN_NAMES, "flag_names": FLAG_NAMES, "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
            "secondary_display_min": SECONDARY_DISPLAY_MIN, "n_train": int(len(tr_df)),
            "epochs_head": args.epochs_head, "epochs_finetune": args.epochs_finetune,
            "lambda_main": args.lambda_main, "holdout": args.holdout, "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out / 'model.pt'} and {out / 'meta.json'}")

    if ho_df is not None:
        probs_main, _ = predict_arrays(model, device, ho_df["path"].tolist(), args.batch_size, args.workers)
        keep = ho_df["main"].to_numpy() != IGNORE
        y, p = ho_df["main"].to_numpy()[keep], probs_main.argmax(1)[keep]
        print("\nHoldout main-stress report (leaves with a predominant label):")
        print(classification_report(y, p, labels=list(range(len(MAIN_NAMES))), target_names=MAIN_NAMES,
                                    digits=3, zero_division=0))


# --------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_arrays(model, device, paths, batch_size=32, workers=0):
    """Return (main-stress softmax probs [N, 5], flag sigmoid probs [N, 4]) for a list of image paths."""
    df = pd.DataFrame({"path": [str(p) for p in paths]})
    for k in FLAG_NAMES:
        df[k] = 0
    df["main"] = 0                                          # dummy targets; only the images are used
    loader = make_loader(df, eval_transform(), batch_size, workers)
    model.eval()
    probs_main, probs_flags = [], []
    for x, _, _ in loader:
        lf, lm = model(x.to(device))
        probs_main.append(torch.softmax(lm, 1).cpu())
        probs_flags.append(torch.sigmoid(lf).cpu())
    return torch.cat(probs_main).numpy(), torch.cat(probs_flags).numpy()


def describe_prediction(main_probs, flag_probs, secondary_min=SECONDARY_DISPLAY_MIN):
    """Turn raw probabilities for one leaf into the reported result.

    The main stress is the argmax of the main head. "Possible additional" stresses are the other stressors
    whose flag probability is at least `secondary_min`; they are shown with their raw probability, never as
    a hard yes/no, and a healthy leaf gets none. Each one also carries the main head's own probability for
    that stressor (`main_head`), and `conflict` is True when a flag is at least as confident as the main
    stress, i.e. the two heads disagree.
    """
    k = int(np.argmax(main_probs))
    result = {"main_stress": MAIN_NAMES[k], "main_probability": float(main_probs[k]),
              "possible_additional": [], "conflict": False}
    if k != 0:
        for j, name in enumerate(FLAG_NAMES):
            if name != MAIN_NAMES[k] and flag_probs[j] >= secondary_min:
                result["possible_additional"].append({"stress": name, "probability": float(flag_probs[j]),
                                                      "main_head": float(main_probs[MAIN_NAMES.index(name)])})
        result["possible_additional"].sort(key=lambda d: -d["probability"])
        result["conflict"] = any(d["probability"] >= result["main_probability"] for d in result["possible_additional"])
    return result


def format_prediction(path, res):
    """One-line text summary of a `describe_prediction` result."""
    s = f"{path}: {res['main_stress']} ({res['main_probability']:.0%})"
    if res["possible_additional"]:
        extra = ", ".join(f"{d['stress']} {d['probability']:.0%} (main head {d['main_head']:.0%})"
                          for d in res["possible_additional"])
        s += (f"  | heads disagree, check manually: {extra}" if res["conflict"]
              else f"  | possible additional stress (low confidence): {extra}")
    return s


# --------------------------------------------------------------------------- heatmaps (Grad-CAM)
def gradcam(model, device, image_path, kind, idx):
    """Grad-CAM for one image and one model output.

    `kind` is "main" (main-stress head) or "flag" (flags head) and `idx` the output index in that head.
    Returns (PIL image, cam scaled to [0, 1] at the feature-map resolution, e.g. 7x7).
    """
    img = Image.open(image_path).convert("RGB")
    x = eval_transform()(img).unsqueeze(0).to(device)
    model.eval()
    model.zero_grad()
    with torch.enable_grad():
        feats = model.features(x)                                   # last convolutional features [1, C, h, w]
        feats.retain_grad()
        h = model.trunk(model.avgpool(feats).flatten(1))
        out = model.head_main(h) if kind == "main" else model.head_flags(h)
        out[0, idx].backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)         # importance of each feature map
        cam = torch.relu((weights * feats).sum(dim=1))[0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return img, cam


def render_heatmap(img, cam, title, out_path, strength=1.0):
    """Save a side-by-side figure: the original leaf and the Grad-CAM overlay.

    The map is normalized per image, so `strength` (the target's probability, 0-1) scales it: an unlikely
    class gives a dim overlay instead of a misleading bright one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heat = Image.fromarray((np.clip(cam * strength, 0, 1) * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(img)
    axes[0].set_title("leaf")
    axes[1].imshow(img)
    axes[1].imshow(np.asarray(heat) / 255.0, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title(title, fontsize=10)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_heatmap(model, device, path, res, flag_probs, target, out_dir):
    """Write the Grad-CAM figure for one leaf into `out_dir` and return its path.

    `target` is "main" (the predicted main stress) or a stressor name (where that flag fires).
    """
    if target == "main":
        kind, idx, what, score = "main", MAIN_NAMES.index(res["main_stress"]), res["main_stress"], res["main_probability"]
    else:
        idx = FLAG_NAMES.index(target)
        kind, what, score = "flag", f"{target} (flag)", float(flag_probs[idx])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img, cam = gradcam(model, device, path, kind, idx)
    out_path = out_dir / f"{Path(path).stem}_gradcam_{what.split()[0]}.png"
    title = (f"Grad-CAM for: {what}, score {score:.0%}\n"
             f"prediction: {res['main_stress']} ({res['main_probability']:.0%})")
    render_heatmap(img, cam, title, out_path, strength=score)
    return out_path


def predict(args):
    """Predict for the images and folders in `args.inputs`; optionally save heatmaps and a CSV."""
    device = pick_device()
    model, meta = load_model(args.model_dir, device)
    paths = []
    for item in args.inputs:
        p = Path(item)
        paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    if not paths:
        raise SystemExit("No images found.")
    probs_main, probs_flags = predict_arrays(model, device, paths, workers=args.workers)
    rows = []
    for path, pm, pf in zip(paths, probs_main, probs_flags):
        res = describe_prediction(pm, pf, meta.get("secondary_display_min", SECONDARY_DISPLAY_MIN))
        print(format_prediction(path, res))
        if args.heatmap_dir:
            print("   heatmap:", save_heatmap(model, device, path, res, pf, args.heatmap_for, args.heatmap_dir))
        rows.append({"path": str(path),
                     **{f"main_{n}": float(pm[i]) for i, n in enumerate(MAIN_NAMES)},
                     **{f"flag_{n}": float(pf[i]) for i, n in enumerate(FLAG_NAMES)},
                     "main_stress": res["main_stress"],
                     "conflict": res["conflict"],
                     "possible_additional": "; ".join(f"{d['stress']}:{d['probability']:.2f}"
                                                      for d in res["possible_additional"])})
    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print("Saved", args.out_csv)


def evaluate(args):
    """Score a trained model on labeled leaves from the CSV (all of them, or only the ids in --ids).

    Prints a main-stress classification report and confusion matrix (leaves coded "multiple" are excluded)
    and per-flag precision/recall/F1 at a 0.5 threshold. Warns when the leaves were part of the training data.
    """
    device = pick_device()
    model, meta = load_model(args.model_dir, device)
    df = load_table(args.csv, args.images)
    if args.ids:
        df = df[df["id"].isin(set(pd.read_csv(args.ids)["id"]))].reset_index(drop=True)
        print(f"Evaluating on {len(df)} leaves listed in {args.ids}")
        if df.empty:
            raise SystemExit("None of the ids in --ids matched a leaf with an image. Check --csv, --images and --ids.")
    if meta.get("holdout", 0) == 0:
        print("\nNOTE: this model was trained on ALL leaves, so these results are on its own training data and "
              "will look better than they would on new leaves. Use the cross-validation numbers as the honest "
              "estimate, or train with --holdout and pass that run's holdout_ids.csv via --ids.\n")
    elif not args.ids:
        print("\nNOTE: no --ids given, so this includes the leaves the model trained on. "
              f"For an unbiased score use --ids {Path(args.model_dir) / 'holdout_ids.csv'}.\n")
    probs_main, probs_flags = predict_arrays(model, device, df["path"].tolist(), workers=args.workers)

    keep = df["main"].to_numpy() != IGNORE
    y, p = df["main"].to_numpy()[keep], probs_main.argmax(1)[keep]
    labels = list(range(len(MAIN_NAMES)))
    print(f"Main stress ({int(keep.sum())} leaves with a predominant label; 'multiple' leaves excluded):")
    print(classification_report(y, p, labels=labels, target_names=MAIN_NAMES, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(y, p, labels=labels), index=MAIN_NAMES, columns=MAIN_NAMES).to_string())

    true_flags = (df[FLAG_NAMES].to_numpy() > 0).astype(int)
    pred_flags = (probs_flags >= 0.5).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(true_flags, pred_flags, average=None, zero_division=0)
    print("\nFlags at 0.5 (all leaves, 'multiple' included):")
    print(pd.DataFrame({"precision": pr, "recall": rc, "f1": f1, "support": sup}, index=FLAG_NAMES).round(3).to_string())

    if args.out_csv:
        out = pd.DataFrame({"id": df["id"], "true_code": df["code"],
                            "pred_main": [MAIN_NAMES[i] for i in probs_main.argmax(1)]})
        for i, n in enumerate(MAIN_NAMES):
            out[f"main_{n}"] = probs_main[:, i]
        for i, n in enumerate(FLAG_NAMES):
            out[f"flag_{n}"] = probs_flags[:, i]
        out.to_csv(args.out_csv, index=False)
        print("Saved", args.out_csv)


# --------------------------------------------------------------------------- command line
def main():
    """Parse the command line and run the chosen command (train, predict or evaluate)."""
    default_workers = 0 if os.name == "nt" else 2       # worker processes are fragile on Windows
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the model")
    t.add_argument("--csv", default="data/leaf/dataset.csv")
    t.add_argument("--images", default="data/leaf/images")
    t.add_argument("--out", default="outputs")
    t.add_argument("--epochs-head", type=int, default=3)
    t.add_argument("--epochs-finetune", type=int, default=15)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--workers", type=int, default=default_workers)
    t.add_argument("--holdout", type=float, default=0.0, help="fraction held out for a sanity check (0 = train on everything)")
    t.add_argument("--lambda-main", type=float, default=0.4, help="weight of the main-stress loss")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=train)

    p = sub.add_parser("predict", help="predict for images or folders")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--model-dir", default="outputs")
    p.add_argument("--workers", type=int, default=default_workers)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--heatmap-dir", default=None,
                   help="also save a Grad-CAM heatmap image for each leaf into this folder")
    p.add_argument("--heatmap-for", default="main", choices=["main"] + FLAG_NAMES,
                   help="'main' = the predicted main stress (default); or a stressor name to see where that flag fires")
    p.set_defaults(func=predict)

    e = sub.add_parser("evaluate", help="score a trained model on labeled leaves from the CSV")
    e.add_argument("--csv", default="data/leaf/dataset.csv")
    e.add_argument("--images", default="data/leaf/images")
    e.add_argument("--model-dir", default="outputs")
    e.add_argument("--ids", default=None, help="CSV with an 'id' column: only evaluate these leaves (e.g. holdout_ids.csv)")
    e.add_argument("--workers", type=int, default=default_workers)
    e.add_argument("--out-csv", default=None)
    e.set_defaults(func=evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()