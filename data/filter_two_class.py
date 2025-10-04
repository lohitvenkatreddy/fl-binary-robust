import torch, torchvision as tv
from torchvision import transforms as T
import json, random, os
import numpy as np
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", type=int, nargs=2, required=True)  # e.g., 1 9
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--out", type=str, default="data/splits.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.RandomState(args.seed)

    transform = T.Compose([T.ToTensor()])
    train = tv.datasets.CIFAR10(root="data/raw", train=True, download=True, transform=transform)
    test = tv.datasets.CIFAR10(root="data/raw", train=False, download=True, transform=transform)

    def filter_indices(ds):
        idx = [i for i, (_, y) in enumerate(ds) if y in args.classes]
        return np.array(idx, dtype=int)

    train_idx = filter_indices(train)
    test_idx  = filter_indices(test)

    # small balanced server validation from train
    y_train = np.array([train[i][1] for i in train_idx])
    cls0, cls1 = args.classes
    c0 = train_idx[y_train == cls0]
    c1 = train_idx[y_train == cls1]
    n0 = max(1, int(len(c0) * args.val_frac))
    n1 = max(1, int(len(c1) * args.val_frac))
    val_idx = np.concatenate([c0[:n0], c1[:n1]])
    keep = np.setdiff1d(train_idx, val_idx, assume_unique=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "classes": args.classes,
            "train_idx": keep.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "seed": args.seed
        }, f)
    print(f"Saved splits to {args.out}")

if __name__ == "__main__":
    main()
