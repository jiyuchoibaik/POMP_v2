# POMP_v2/pre-training/dataset_loader.py
"""
POMP 데이터셋
────────────────────────────────────────────────────────
디렉토리 구조:
  preprocessed/
      wsi/TCGA-XX-XXXX/patches.pt          (N, 3, 256, 256) float32 [0,1]
      rna/TCGA-XX-XXXX/pathway_scores.pt   (50,) float32

동작:
  - WSI + RNA 둘 다 있는 케이스만 포함
  - exclusion list (wsi_few_patch_cases.txt 등) 자동 적용
  - WSI 패치 수 N이 다른 케이스: n_patches개 랜덤 샘플링
  - N < n_patches면 반복 샘플링 (over-sampling)
"""

import os
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class POMPDataset(Dataset):
    def __init__(
        self,
        wsi_dir:          str,
        rna_dir:          str,
        n_patches:        int  = 100,    # 환자당 샘플링할 패치 수
        split:            str  = "train",
        val_ratio:        float = 0.1,
        seed:             int  = 42,
        exclusion_files:  list = None,   # 제외할 case_id 목록 파일 경로들
    ):
        """
        Args:
            wsi_dir:         preprocessed/wsi 디렉토리
            rna_dir:         preprocessed/rna 디렉토리
            n_patches:       환자당 샘플링할 패치 수
            split:           "train" | "val" | "all"
            val_ratio:       validation 비율
            seed:            랜덤 시드 (train/val split 재현성)
            exclusion_files: 제외할 case_id 파일 경로 리스트
                             예) ["preprocessed/wsi/wsi_few_patch_cases.txt"]
        """
        self.wsi_dir   = Path(wsi_dir)
        self.rna_dir   = Path(rna_dir)
        self.n_patches = n_patches

        # ── 제외 케이스 로드 ──────────────────────────────────────────────
        excluded = set()
        if exclusion_files:
            for fpath in exclusion_files:
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        for line in f:
                            cid = line.strip()
                            if cid:
                                excluded.add(cid)
                    print(f"[Dataset] 제외 목록 로드: {fpath} ({len(excluded)}개)")

        # ── WSI ∩ RNA 케이스 탐색 ─────────────────────────────────────────
        wsi_cases = {
            d.name for d in self.wsi_dir.iterdir()
            if d.is_dir() and (d / "patches.pt").exists()
        }
        rna_cases = {
            d.name for d in self.rna_dir.iterdir()
            if d.is_dir() and (d / "pathway_scores.pt").exists()
        }

        valid_cases = sorted((wsi_cases & rna_cases) - excluded)
        print(f"[Dataset] WSI: {len(wsi_cases)}  RNA: {len(rna_cases)}  "
              f"교집합: {len(wsi_cases & rna_cases)}  "
              f"제외 후: {len(valid_cases)}")

        # ── train / val split ─────────────────────────────────────────────
        rng = random.Random(seed)
        shuffled = valid_cases[:]
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        val_cases   = set(shuffled[:n_val])
        train_cases = set(shuffled[n_val:])

        if split == "train":
            self.cases = sorted(train_cases)
        elif split == "val":
            self.cases = sorted(val_cases)
        else:  # "all"
            self.cases = valid_cases

        print(f"[Dataset] split={split}  케이스 수: {len(self.cases)}")

        # ── 패치 수 사전 확인 ─────────────────────────────────────────────
        self._patch_counts = {}
        for cid in self.cases:
            pt = torch.load(
                self.wsi_dir / cid / "patches.pt",
                map_location="cpu",
            )
            self._patch_counts[cid] = pt.shape[0]

        counts = list(self._patch_counts.values())
        if len(counts) > 0:
            print(f"[Dataset] 패치 수  min={min(counts)}  "
                f"max={max(counts)}  mean={np.mean(counts):.1f}")
        else:
            print("[Dataset] empty dataset")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        cid = self.cases[idx]

        # ── WSI 패치 로드 ─────────────────────────────────────────────────
        patches = torch.load(
            self.wsi_dir / cid / "patches.pt",
            map_location="cpu",
        )                                     # (N, 3, 256, 256) float32

        N = patches.shape[0]
        if N >= self.n_patches:
            # 랜덤 서브샘플링
            sel = torch.randperm(N)[: self.n_patches]
        else:
            # over-sampling (N < n_patches)
            sel = torch.randint(0, N, (self.n_patches,))
        patches = patches[sel]                # (n_patches, 3, 256, 256)

        # ── RNA 경로 점수 로드 ────────────────────────────────────────────
        pathway_scores = torch.load(
            self.rna_dir / cid / "pathway_scores.pt",
            map_location="cpu",
        )                                     # (50,) float32

        return {
            "patches":        patches,         # (n_patches, 3, 256, 256)
            "pathway_scores": pathway_scores,  # (50,)
            "case_id":        cid,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Collate (case_id 포함)
# ══════════════════════════════════════════════════════════════════════════════
def pomp_collate_fn(batch: list) -> dict:
    """
    n_patches가 고정이므로 패치는 단순 stack.
    case_id는 문자열 리스트로 반환.
    """
    patches        = torch.stack([b["patches"]        for b in batch])  # (B, N, 3, 256, 256)
    pathway_scores = torch.stack([b["pathway_scores"] for b in batch])  # (B, 50)
    case_ids       = [b["case_id"] for b in batch]

    return {
        "patches":        patches,
        "pathway_scores": pathway_scores,
        "case_ids":       case_ids,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader 빌더
# ══════════════════════════════════════════════════════════════════════════════
def build_loaders(
    wsi_dir:          str,
    rna_dir:          str,
    n_patches:        int   = 100,
    batch_size:       int   = 8,
    num_workers:      int   = 4,
    val_ratio:        float = 0.1,
    seed:             int   = 42,
    exclusion_files:  list  = None,
) -> tuple:
    """
    Returns: (train_loader, val_loader)
    """
    train_ds = POMPDataset(
        wsi_dir=wsi_dir, rna_dir=rna_dir,
        n_patches=n_patches,
        split="train", val_ratio=val_ratio,
        seed=seed, exclusion_files=exclusion_files,
    )
    val_ds = POMPDataset(
        wsi_dir=wsi_dir, rna_dir=rna_dir,
        n_patches=n_patches,
        split="val", val_ratio=val_ratio,
        seed=seed, exclusion_files=exclusion_files,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pomp_collate_fn,
        drop_last=True,     # ITC loss: 배치 크기 일정해야 함
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pomp_collate_fn,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    print(f"\n[DataLoader]")
    print(f"  train: {len(train_ds)}명  {len(train_loader)} batches")
    print(f"  val:   {len(val_ds)}명   {len(val_loader)} batches")
    print(f"  batch_size={batch_size}  n_patches={n_patches}")

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_loader, val_loader = build_loaders(
        wsi_dir="../datasets/preprocessed/wsi",
        rna_dir="../datasets/preprocessed/rna",
        n_patches=100,
        batch_size=4,
        num_workers=2,
        exclusion_files=[
            "../datasets/preprocessed/wsi/wsi_few_patch_cases.txt",
            "../datasets/preprocessed/wsi/wsi_skipped_cases.txt",
        ],
    )

    batch = next(iter(train_loader))
    print(f"\n배치 확인:")
    print(f"  patches:        {batch['patches'].shape}")         # (4, 100, 3, 256, 256)
    print(f"  pathway_scores: {batch['pathway_scores'].shape}")  # (4, 50)
    print(f"  case_ids:       {batch['case_ids']}")