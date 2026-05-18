import re

log_file = "./checkpoints/train_20260514_165844.log"

results = []

# Epoch [  4/200]  train_loss=3.2512  val_loss=3.0460
pattern = re.compile(
    r"Epoch\s+\[\s*(\d+)/\d+\].*?train_loss=([\d.]+)\s+val_loss=([\d.]+)"
)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))

            results.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

# val_loss 기준 오름차순 정렬
top5 = sorted(results, key=lambda x: x["val_loss"])[:5]

print("\n=== Top 5 Lowest val_loss Epochs ===\n")

for rank, item in enumerate(top5, start=1):
    print(
        f"{rank}. Epoch {item['epoch']:3d} | "
        f"train_loss={item['train_loss']:.4f} | "
        f"val_loss={item['val_loss']:.4f}"
    )