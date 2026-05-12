"""
패치 수 20개 미만 WSI 및 대응 RNA 삭제 스크립트
- 20개 미만 WSI + 대응 RNA 삭제
- WSI는 없는데 RNA만 있는 케이스도 삭제
"""
import os
import shutil
import torch
from pathlib import Path

WSI_DIR = "./preprocessed/wsi"
RNA_DIR = "./preprocessed/rna"
MIN_PATCHES = 20

wsi_root = Path(WSI_DIR)
rna_root = Path(RNA_DIR)

removed_wsi = []
removed_rna = []
orphan_rna = []  # WSI 없는 RNA
skipped = []

# ── 1. 패치 수 미달 WSI + 대응 RNA 삭제 ──────────────────────────────
case_dirs = sorted([d for d in wsi_root.iterdir() if d.is_dir()])
print(f"전체 WSI 케이스: {len(case_dirs)}명")

for case_dir in case_dirs:
    patches_path = case_dir / "patches.pt"
    case_id = case_dir.name

    if not patches_path.exists():
        skipped.append(case_id)
        continue

    patches = torch.load(patches_path, weights_only = True)
    n_patches = patches.shape[0]

    if n_patches < MIN_PATCHES:
        shutil.rmtree(case_dir)
        removed_wsi.append((case_id, n_patches))

        rna_case_dir = rna_root / case_id
        if rna_case_dir.exists():
            shutil.rmtree(rna_case_dir)
            removed_rna.append(case_id)
        else:
            print(f"  [INFO] 대응 RNA 없음 (이미 삭제됐거나 원래 없음): {case_id}")

# ── 2. WSI 없는 고아 RNA 삭제 ─────────────────────────────────────────
print(f"\n고아 RNA 탐색 중...")
rna_case_dirs = sorted([d for d in rna_root.iterdir() if d.is_dir()])

for rna_case_dir in rna_case_dirs:
    case_id = rna_case_dir.name
    wsi_case_dir = wsi_root / case_id

    if not wsi_case_dir.exists():
        shutil.rmtree(rna_case_dir)
        orphan_rna.append(case_id)
        print(f"  [삭제] 고아 RNA: {case_id}")

# ── 요약 ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  제거된 WSI (패치 미달) : {len(removed_wsi)}명")
print(f"  제거된 RNA (WSI 미달)  : {len(removed_rna)}명")
print(f"  제거된 고아 RNA        : {len(orphan_rna)}명")
print(f"  patches.pt 없어 스킵   : {len(skipped)}명")
print(f"  최종 남은 WSI          : {len(case_dirs) - len(removed_wsi) - len(skipped)}명")
print(f"{'='*50}")

if removed_wsi:
    print(f"\n패치 미달 제거 목록 ({MIN_PATCHES}개 미만):")
    for case_id, n in removed_wsi:
        print(f"  {case_id}: {n}개 패치")

if orphan_rna:
    print(f"\n고아 RNA 제거 목록:")
    for case_id in orphan_rna:
        print(f"  {case_id}")