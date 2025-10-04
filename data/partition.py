import json, argparse, numpy as np, os

def dirichlet_partition(labels, num_clients, alpha, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    idx_by_c = {c: np.where(labels == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_c[c])
    proportions = rng.dirichlet(alpha=[alpha]*num_clients, size=len(classes))
    client_indices = [[] for _ in range(num_clients)]
    for ci, c in enumerate(classes):
        counts = (proportions[ci] / proportions[ci].sum() * len(idx_by_c[c])).astype(int)
        # fix rounding
        while counts.sum() < len(idx_by_c[c]): counts[rng.randint(0, num_clients)] += 1
        start = 0
        for k in range(num_clients):
            end = start + counts[k]
            client_indices[k].extend(idx_by_c[c][start:end].tolist())
            start = end
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices

def iid_partition(num_samples, num_clients, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    return np.array_split(idx, num_clients)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)         # data/splits.json
    ap.add_argument("--scheme", choices=["iid","dirichlet"], default="dirichlet")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--num-clients", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = json.load(open(args.splits))
    train_idx = np.array(meta["train_idx"])
    # assume labels available separately or regenerate via dataset when training
    # here we only shard indices uniformly; label-aware dirichlet typically needs labels
    if args.scheme == "iid":
        shards = [arr.tolist() for arr in np.array_split(train_idx, args.num_clients)]
    else:
        raise SystemExit("Label-aware Dirichlet will be computed in training where labels are accessible.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"clients": shards, "scheme": args.scheme, "seed": args.seed}, open(args.out, "w"))
    print(f"Wrote partition to {args.out}")

if __name__ == "__main__":
    main()
