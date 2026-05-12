# POMP_v2/pre-training/train.py
"""
POMP 사전학습 루프
────────────────────────────────────────────────────────
특징:
  - Mixed precision (torch.cuda.amp)
  - Cosine LR + warmup
  - Gradient clipping
  - Hard negative curriculum (pomp.py에서 관리)
  - WandB 로깅 (선택)
  - 체크포인트 저장/재개

사용법:
  # 기본 학습
  python train.py \
      --wsi_dir ../datasets/preprocessed/wsi \
      --rna_dir ../datasets/preprocessed/rna \
      --out_dir ./checkpoints \
      --epochs 100 \
      --batch_size 8

  # 체크포인트 재개
  python train.py ... --resume ./checkpoints/last.pt

  # WandB 로깅
  python train.py ... --use_wandb --wandb_project pomp_pretrain
"""

import os
import sys
import math
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# 상위 폴더에서 import
sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader import build_loaders
from pomp import POMPModel


# ══════════════════════════════════════════════════════════════════════════════
# LR Scheduler (cosine + warmup)
# ══════════════════════════════════════════════════════════════════════════════
def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, n_iter_per_ep: int):
    """
    step 단위 cosine LR + linear warmup.
    warmup: 0 → lr
    cosine: lr → lr_min (lr * 0.01)
    """
    warmup_steps  = warmup_epochs * n_iter_per_ep
    total_steps   = total_epochs  * n_iter_per_ep

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(state: dict, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_val    = ckpt.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] {path} 로드 완료 → epoch {start_epoch}부터 재개")
    return start_epoch, best_val


# ══════════════════════════════════════════════════════════════════════════════
# Metric Tracker
# ══════════════════════════════════════════════════════════════════════════════
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: dict, n: int = 1):
        for k, v in metrics.items():
            if k not in self._sums:
                self._sums[k]   = 0.0
                self._counts[k] = 0
            self._sums[k]   += v * n
            self._counts[k] += n

    def avg(self) -> dict:
        return {
            k: self._sums[k] / max(1, self._counts[k])
            for k in self._sums
        }


# ══════════════════════════════════════════════════════════════════════════════
# 단일 epoch 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model:      POMPModel,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    scaler:     GradScaler,
    device:     torch.device,
    epoch:      int,
    args,
) -> dict:

    model.train()
    model.set_epoch(epoch)      # hard negative curriculum 업데이트

    tracker  = MetricTracker()
    n_steps  = len(loader)
    t_start  = time.time()

    for step, batch in enumerate(loader):
        patches        = batch["patches"].to(device, non_blocking=True)
        pathway_scores = batch["pathway_scores"].to(device, non_blocking=True)

        with autocast(enabled=args.amp):
            out = model(patches, pathway_scores, mode="pretrain")
            loss = out["loss"]

        # ── 임시 디버그 (step==0에만) ──────────────────────────────
        if step == 0 and epoch == 1:
            with torch.no_grad():
                cls_p, wsi_tokens = model.wsi_encoder(patches)
                cls_o, rna_tokens = model.rna_encoder(pathway_scores)

                out_pos = model.mm_encoder(
                    wsi_tokens, rna_tokens, cls_p, cls_o
                )
                print(f"\n[DEBUG] itm_logits (pos):\n{out_pos['itm_logits']}")
                print(f"[DEBUG] H_wsi_cls norm: {out_pos['H_wsi_cls'].norm(dim=-1)}")
                print(f"[DEBUG] H_rna_cls norm: {out_pos['H_rna_cls'].norm(dim=-1)}")

                neg_rna   = rna_tokens[torch.randperm(rna_tokens.shape[0])]
                neg_cls_o = cls_o[torch.randperm(cls_o.shape[0])]
                out_neg   = model.mm_encoder(
                    wsi_tokens, neg_rna, cls_p, neg_cls_o
                )
                print(f"[DEBUG] itm_logits (neg):\n{out_neg['itm_logits']}")

                B = patches.shape[0]
                logits = torch.cat([out_pos['itm_logits'], out_neg['itm_logits']], dim=0)
                labels = torch.cat([
                    torch.ones(B, dtype=torch.long),
                    torch.zeros(B, dtype=torch.long),
                ]).to(device)
                print(f"[DEBUG] ITM loss 직접: {F.cross_entropy(logits, labels).item():.4f}")
        # ── 디버그 끝 ─────────────────────────────────────────────
        # gradient accumulation
        loss = loss / args.grad_accum
        scaler.scale(loss).backward()

        is_update_step = (
            (step + 1) % args.grad_accum == 0
            or (step + 1) == n_steps
        )

        if is_update_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        B = patches.shape[0]
        tracker.update({
            "loss":     out["loss"].item(),
            "loss_itc": out["loss_itc"].item(),
            "loss_itm": out["loss_itm"].item(),
            "loss_mom": out["loss_mom"].item(),
        }, n=B)

        # 진행 출력
        if step % args.log_interval == 0:
            lr  = scheduler.get_last_lr()[0]
            avg = tracker.avg()
            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (n_steps - step - 1)
            print(
                f"  Epoch [{epoch:3d}] [{step:4d}/{n_steps}]  "
                f"loss={avg['loss']:.4f}  "
                f"itc={avg['loss_itc']:.4f}  "
                f"itm={avg['loss_itm']:.4f}  "
                f"mom={avg['loss_mom']:.4f}  "
                f"lr={lr:.2e}  ETA={eta:.0f}s"
            )

    return tracker.avg()


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(
    model:  POMPModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    args,
) -> dict:

    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        patches        = batch["patches"].to(device, non_blocking=True)
        pathway_scores = batch["pathway_scores"].to(device, non_blocking=True)

        with autocast(enabled=args.amp):
            out = model(patches, pathway_scores, mode="pretrain")

        B = patches.shape[0]
        tracker.update({
            "loss":     out["loss"].item(),
            "loss_itc": out["loss_itc"].item(),
            "loss_itm": out["loss_itm"].item(),
            "loss_mom": out["loss_mom"].item(),
        }, n=B)

    return tracker.avg()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    # ── 재현성 ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── 디바이스 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── WandB ─────────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or datetime.now().strftime("%Y%m%d_%H%M%S"),
            config=vars(args),
        )

    # ── 데이터로더 ────────────────────────────────────────────────────────
    exclusion_files = [
        os.path.join(args.wsi_dir, "wsi_few_patch_cases.txt"),
        os.path.join(args.wsi_dir, "wsi_skipped_cases.txt"),
    ]
    train_loader, val_loader = build_loaders(
        wsi_dir=args.wsi_dir,
        rna_dir=args.rna_dir,
        n_patches=args.n_patches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        exclusion_files=[f for f in exclusion_files if os.path.exists(f)],
    )

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = POMPModel(
        uni2_local_dir=args.uni2_dir,
        total_epochs=args.epochs,
        hard_negative_start_epoch=args.hard_neg_start,
        itc_weight=args.itc_weight,
        itm_weight=args.itm_weight,
        mom_weight=args.mom_weight,
        temp=args.temp,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 학습 파라미터: {n_params:,}개")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # UNI2는 freeze이므로 학습 파라미터만 optimizer에 전달
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        n_iter_per_ep=len(train_loader),
    )

    # ── AMP Scaler ────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=args.amp)

    # ── 체크포인트 재개 ───────────────────────────────────────────────────
    start_epoch  = 0
    best_val     = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )

    # ── 설정 저장 ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # 학습 루프
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  사전학습 시작: {args.epochs}epochs  batch={args.batch_size}")
    print(f"  hard negative start epoch: {model.hard_negative_start_epoch}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        ep_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args,
        )

        # ── Validation ────────────────────────────────────────────────────
        val_metrics = validate(model, val_loader, device, args)

        ep_time = time.time() - ep_start
        print(
            f"\nEpoch [{epoch:3d}/{args.epochs}]  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"({ep_time:.0f}s)\n"
            f"  train → itc={train_metrics['loss_itc']:.4f}  "
            f"itm={train_metrics['loss_itm']:.4f}  "
            f"mom={train_metrics['loss_mom']:.4f}  "
            f"[ITM {'ON' if epoch >= model.itm_start_epoch else 'OFF'}]\n"  # ← 추가
            f"  val   → itc={val_metrics['loss_itc']:.4f}  "
            f"itm={val_metrics['loss_itm']:.4f}  "
            f"mom={val_metrics['loss_mom']:.4f}"
        )

        # ── WandB 로깅 ────────────────────────────────────────────────────
        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch":          epoch,
                "train/loss":     train_metrics["loss"],
                "train/loss_itc": train_metrics["loss_itc"],
                "train/loss_itm": train_metrics["loss_itm"],
                "train/loss_mom": train_metrics["loss_mom"],
                "val/loss":       val_metrics["loss"],
                "val/loss_itc":   val_metrics["loss_itc"],
                "val/loss_itm":   val_metrics["loss_itm"],
                "val/loss_mom":   val_metrics["loss_mom"],
                "lr":             scheduler.get_last_lr()[0],
            })

        # ── 체크포인트 저장 ────────────────────────────────────────────────
        ckpt = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics":   val_metrics,
            "best_val_loss": best_val,
            "args":          vars(args),
        }

        # last.pt: 항상 저장
        save_checkpoint(ckpt, args.out_dir, "last.pt")

        # best.pt: val_loss 개선 시
        val_loss = val_metrics["loss"]
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt, args.out_dir, "best.pt")
            print(f"  ★ best val_loss 갱신: {best_val:.4f}")

        # 주기적 저장
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(ckpt, args.out_dir, f"epoch_{epoch:03d}.pt")

        print()

    print(f"\n사전학습 완료! best val_loss: {best_val:.4f}")
    print(f"체크포인트 위치: {args.out_dir}")

    if args.use_wandb:
        import wandb
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Argparse
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(description="POMP 사전학습")

    # 데이터
    ap.add_argument("--wsi_dir",     required=True,
                    help="preprocessed/wsi 경로")
    ap.add_argument("--rna_dir",     required=True,
                    help="preprocessed/rna 경로")
    ap.add_argument("--uni2_dir",    default="./assets/uni2",
                    help="UNI2-h 가중치 디렉토리")
    ap.add_argument("--n_patches",   type=int, default=100,
                    help="환자당 샘플링 패치 수 (default: 100)")
    ap.add_argument("--val_ratio",   type=float, default=0.1,
                    help="validation 비율 (default: 0.1)")

    # 학습
    ap.add_argument("--out_dir",     default="./checkpoints",
                    help="체크포인트 저장 경로")
    ap.add_argument("--epochs",      type=int, default=100)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--weight_decay",type=float, default=1e-2)
    ap.add_argument("--warmup_epochs",type=int, default=5)
    ap.add_argument("--clip_grad",   type=float, default=1.0)
    ap.add_argument("--grad_accum",  type=int, default=1,
                    help="gradient accumulation steps")
    ap.add_argument("--seed",        type=int, default=42)

    # Loss 가중치
    ap.add_argument("--itc_weight",  type=float, default=1.0)
    ap.add_argument("--itm_weight",  type=float, default=2.0)
    ap.add_argument("--mom_weight",  type=float, default=1.0)
    ap.add_argument("--temp",        type=float, default=0.07,
                    help="ITC temperature")

    # Hard negative curriculum
    ap.add_argument("--hard_neg_start", type=int, default=None,
                    help="hard negative 시작 epoch (None이면 epoch*0.3)")

    # 기타
    ap.add_argument("--amp",         action="store_true", default=True,
                    help="Mixed precision (default: True)")
    ap.add_argument("--no_amp",      action="store_false", dest="amp")
    ap.add_argument("--resume",      default=None,
                    help="재개할 체크포인트 경로")
    ap.add_argument("--log_interval",type=int, default=10,
                    help="로그 출력 간격 (step 단위)")
    ap.add_argument("--save_interval",type=int, default=10,
                    help="주기적 체크포인트 저장 간격 (epoch 단위)")

    # WandB
    ap.add_argument("--use_wandb",   action="store_true")
    ap.add_argument("--wandb_project", default="pomp_pretrain")
    ap.add_argument("--wandb_run",   default=None)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)