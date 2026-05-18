#!/usr/bin/env python
"""
POMP_v2/pre-training/geometric_analysis.py
─────────────────────────────────────────────────────────────
3개 체크포인트에 대한 기하학적 표현 분석

분석 지표:
  1. CKA (Centered Kernel Alignment)  – 두 모달리티 관계 구조 유사도
  2. Modality Gap                      – 무게중심 간 L2 거리
  3. Intra-modal Uniformity            – hypersphere 상 분포 균등성
  4. Cross-modal Isotropy              – 두 모달리티 분포 형태의 대칭성

분석 표현:
  cls_p / cls_o  : projection 이전 단일모달 CLS 토큰 (raw)
  z_wsi  / z_rna : L2-normalized ITC 표현

체크포인트:
  epoch_009.pt  →  Epoch 9  (초기, ITM 활성화 전)
  best.pt       →  Epoch 14 (val_loss 최저)
  epoch_199.pt  →  Epoch 199 (학습 완료)

사용법:
  python geometric_analysis.py \\
      --wsi_dir  ../datasets/preprocessed/wsi \\
      --rna_dir  ../datasets/preprocessed/rna \\
      --ckpt_dir ./checkpoints \\
      --uni2_dir ./assets/uni2

출력:
  터미널 결과 테이블 + geometric_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from pomp import POMPModel


# ══════════════════════════════════════════════════════════════════════════════
# 패치 인덱스 사전 고정
# ══════════════════════════════════════════════════════════════════════════════

def precompute_patch_indices(
    patient_ids: List[str],
    wsi_dir: Path,
    n_patches: int,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    모든 환자의 패치 샘플링 인덱스를 사전에 고정.

    체크포인트마다 collect_embeddings를 반복 호출할 때
    동일한 패치가 선택되도록 보장합니다.

    파일시스템 순서 변동이나 체크포인트 간 RNG 상태 차이에 무관하게
    재현 가능한 비교를 위해 모든 체크포인트 루프 진입 전 1회만 호출합니다.

    Returns:
        {patient_id: LongTensor(n_patches)}  – 각 환자의 패치 인덱스
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    indices: Dict[str, torch.Tensor] = {}

    for pid in patient_ids:
        patches_path = wsi_dir / pid / "patches.pt"

        try:
            # shape 확인만을 위해 헤더 수준으로 로드
            meta = torch.load(patches_path, map_location="cpu", weights_only=True)
        except TypeError:
            meta = torch.load(patches_path, map_location="cpu")

        N_avail = meta.shape[0]
        del meta  # 메모리 즉시 해제

        if N_avail >= n_patches:
            idx = torch.randperm(N_avail, generator=rng)[:n_patches]
        else:
            # 패치 수 부족 시 over-sampling (중복 허용)
            idx = torch.randint(0, N_avail, (n_patches,), generator=rng)

        indices[pid] = idx  # LongTensor(n_patches)

    return indices


# ══════════════════════════════════════════════════════════════════════════════
# 기하학적 지표 함수
# ══════════════════════════════════════════════════════════════════════════════

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA (Kornblith et al., 2019)

    두 표현 행렬의 관계 구조(pairwise similarity 패턴) 유사도.
    범위: [0, 1]  →  높을수록 두 모달리티가 같은 관계 구조를 가짐.

    수식: CKA(K, L) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F · ||Yc^T Yc||_F)
          Xc = X - mean(X, dim=0)  (샘플 평균 제거)

    Args:
        X: (N, d1)
        Y: (N, d2)
    """
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)

    dot_xy = (X.T @ Y).pow(2).sum()
    dot_xx = (X.T @ X).pow(2).sum()
    dot_yy = (Y.T @ Y).pow(2).sum()

    denom = dot_xx.sqrt() * dot_yy.sqrt()
    if denom < 1e-12:
        return 0.0
    return (dot_xy / denom).item()


def modality_gap(Z_wsi: torch.Tensor, Z_rna: torch.Tensor) -> float:
    """
    Modality Gap (Liang et al., 2022)

    두 모달리티 임베딩 무게중심 간 L2 거리.
    범위: [0, ∞)  →  낮을수록 두 모달리티가 같은 공간에 위치.

    Args:
        Z_wsi: (N, d)
        Z_rna: (N, d)
    """
    return (Z_wsi.float().mean(0) - Z_rna.float().mean(0)).norm().item()


def uniformity(Z: torch.Tensor, t: float = 2.0, prenormalized: bool = False) -> float:
    """
    Intra-modal Uniformity (Wang & Isola, 2020)

    단일 모달리티 임베딩의 hypersphere 상 균등 분포 정도.
    범위: (-∞, 0]  →  낮을수록 더 균등하게 분포 (이상적).

    수식: log E[exp(-t · ||z_i - z_j||^2)]  (pairwise 평균)

    정규화 정책:
      - cls_p / cls_o (raw CLS): prenormalized=False → 내부에서 L2 정규화 후 계산
      - z_wsi / z_rna (ITC 출력): prenormalized=True  → 이미 normalized이므로 skip

    두 표현 간 uniformity를 비교하려면 동일한 hypersphere 위에 있어야 하므로,
    cls 표현도 반드시 정규화 후 계산합니다.
    prenormalized 플래그는 "불필요한 이중 normalize를 피하기 위한" 최적화이며,
    의미론적 차이를 만들지 않습니다.

    Args:
        Z:            (N, d)
        t:            RBF 커널 bandwidth (기본값 2.0)
        prenormalized: True이면 이미 unit-norm → normalize skip
    """
    if not prenormalized:
        Z = F.normalize(Z.float(), dim=-1)
    else:
        Z = Z.float()  # dtype 통일만

    sq_pdist = torch.pdist(Z, p=2).pow(2)  # N*(N-1)/2 개 거리
    return sq_pdist.mul(-t).exp().mean().log().item()


def isotropy_score(Z: torch.Tensor) -> float:
    """
    Intra-modal Isotropy

    단일 모달리티 분포의 등방성 (모든 방향으로 균등히 퍼져있는 정도).
    범위: (0, 1]  →  1에 가까울수록 완전 등방 (균등한 공분산).

    수식: I(Z) = d * λ_min / Σλ_i
          λ: 공분산 행렬의 고유값 (오름차순 정렬)
          d: 임베딩 차원

    λ_min/mean이 1/d에 가까울수록 등방적.

    Args:
        Z: (N, d)
    """
    Z = Z.float() - Z.float().mean(0)
    C = (Z.T @ Z) / max(len(Z) - 1, 1)                  # (d, d) 공분산
    eigvals = torch.linalg.eigvalsh(C).clamp(min=0)      # 오름차순

    total = eigvals.sum()
    if total < 1e-12:
        return 0.0
    d = eigvals.shape[0]
    return (d * eigvals.min() / total).item()


def cross_modal_isotropy(Z_wsi: torch.Tensor, Z_rna: torch.Tensor) -> float:
    """
    Cross-modal Isotropy

    두 모달리티 공분산 구조의 대칭성.
    각 모달리티의 정규화된 고유값 스펙트럼 간 Pearson 상관계수.
    범위: [-1, 1]  →  1에 가까울수록 두 분포 형태가 대칭적.

    고유값 스펙트럼 = 공분산 행렬의 주축(principal axis) 분포를 요약.
    두 모달리티가 같은 '형태'로 분포할수록 상관이 높음.

    Args:
        Z_wsi: (N, d)
        Z_rna: (N, d)
    """
    def _normalized_eigvals(Z: torch.Tensor) -> np.ndarray:
        Z = Z.float() - Z.float().mean(0)
        C = (Z.T @ Z) / max(len(Z) - 1, 1)
        ev = torch.linalg.eigvalsh(C).clamp(min=0)
        total = ev.sum()
        if total < 1e-12:
            return ev.cpu().numpy()
        return (ev / total).cpu().numpy()

    ev_wsi = _normalized_eigvals(Z_wsi)
    ev_rna = _normalized_eigvals(Z_rna)

    # 차원이 다를 경우 큰 고유값(주축) 기준으로 맞춤
    # eigvalsh는 오름차순이므로 [-min_d:] 로 상위 주축만 비교
    min_d = min(len(ev_wsi), len(ev_rna))
    ev_wsi = ev_wsi[-min_d:]
    ev_rna = ev_rna[-min_d:]

    corr = np.corrcoef(ev_wsi, ev_rna)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_all_metrics(
    cls_p: torch.Tensor,
    cls_o: torch.Tensor,
    z_wsi: torch.Tensor,
    z_rna: torch.Tensor,
) -> Dict[str, float]:
    """
    4가지 기하학적 지표를 cls 표현과 z 표현 각각에 대해 계산.

    uniformity 호출 시 prenormalized 플래그:
      - cls_p / cls_o: raw CLS → False (함수 내부에서 정규화)
      - z_wsi / z_rna: ITC 출력, 이미 unit-norm → True (정규화 skip)

    Returns:
        dict:
            cls/cka, cls/modality_gap,
            cls/uniformity_wsi, cls/uniformity_rna,
            cls/isotropy_wsi, cls/isotropy_rna, cls/cross_modal_isotropy
            z/cka, z/modality_gap,
            z/uniformity_wsi, z/uniformity_rna,
            z/isotropy_wsi, z/isotropy_rna, z/cross_modal_isotropy
    """
    results: Dict[str, float] = {}

    # (태그, WSI 표현, RNA 표현, 이미 정규화 여부)
    configs = [
        ("cls", cls_p.cpu(), cls_o.cpu(), False),  # raw CLS → normalize 필요
        ("z",   z_wsi.cpu(), z_rna.cpu(), True),   # ITC 출력 → 이미 unit-norm
    ]

    for tag, W, R, is_prenorm in configs:
        results[f"{tag}/cka"]                  = linear_cka(W, R)
        results[f"{tag}/modality_gap"]         = modality_gap(W, R)
        results[f"{tag}/uniformity_wsi"]       = uniformity(W, prenormalized=is_prenorm)
        results[f"{tag}/uniformity_rna"]       = uniformity(R, prenormalized=is_prenorm)
        results[f"{tag}/isotropy_wsi"]         = isotropy_score(W)
        results[f"{tag}/isotropy_rna"]         = isotropy_score(R)
        results[f"{tag}/cross_modal_isotropy"] = cross_modal_isotropy(W, R)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로딩
# ══════════════════════════════════════════════════════════════════════════════

def find_patients(wsi_dir: Path, rna_dir: Path) -> List[str]:
    """WSI와 RNA 데이터가 모두 존재하는 환자 ID 목록을 반환."""
    wsi_ids = {
        p.name for p in wsi_dir.iterdir()
        if p.is_dir() and (p / "patches.pt").exists()
    }
    rna_ids = {
        p.name for p in rna_dir.iterdir()
        if p.is_dir() and (p / "pathway_scores.pt").exists()
    }
    common = sorted(wsi_ids & rna_ids)
    print(f"  WSI {len(wsi_ids)}명 / RNA {len(rna_ids)}명 → 공통 {len(common)}명")
    return common


def load_patient(
    patient_id: str,
    wsi_dir: Path,
    rna_dir: Path,
    patch_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    단일 환자 데이터 로드 및 패치 샘플링.

    Args:
        patch_idx: precompute_patch_indices 에서 사전 고정된 LongTensor(n_patches)

    Returns:
        patches:        (1, n_patches, 3, 256, 256)
        pathway_scores: (1, 50)
    """
    # WSI 패치 로드
    try:
        patches = torch.load(
            wsi_dir / patient_id / "patches.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        patches = torch.load(
            wsi_dir / patient_id / "patches.pt",
            map_location="cpu",
        )

    # 사전 고정된 인덱스로 패치 선택 (체크포인트 간 동일 패치 보장)
    patches = patches[patch_idx].unsqueeze(0)  # (1, n_patches, 3, 256, 256)

    # RNA pathway scores 로드
    try:
        pathway_scores = torch.load(
            rna_dir / patient_id / "pathway_scores.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        pathway_scores = torch.load(
            rna_dir / patient_id / "pathway_scores.pt",
            map_location="cpu",
        )

    pathway_scores = pathway_scores.unsqueeze(0)  # (1, 50)

    return patches, pathway_scores


@torch.no_grad()
def collect_embeddings(
    model: POMPModel,
    patient_ids: List[str],
    wsi_dir: Path,
    rna_dir: Path,
    patch_indices: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    모든 환자에 대해 4종 임베딩을 수집.

    패치 인덱스는 precompute_patch_indices()에서 사전 고정된 값을 사용합니다.
    체크포인트가 달라도 동일한 패치를 참조하므로 공정한 비교가 보장됩니다.

    Args:
        patch_indices: {patient_id: LongTensor(n_patches)}

    Returns:
        {
            "cls_p": (N, 256),
            "cls_o": (N, 256),
            "z_wsi": (N, 256),
            "z_rna": (N, 256),
        }
    """
    model.eval()

    all_cls_p, all_cls_o, all_z_wsi, all_z_rna = [], [], [], []

    for i, pid in enumerate(patient_ids):
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == len(patient_ids):
            print(f"    [{i+1:3d}/{len(patient_ids)}] {pid}")

        patches, pathway_scores = load_patient(
            pid, wsi_dir, rna_dir,
            patch_idx=patch_indices[pid],   # 사전 고정 인덱스 사용
        )
        patches        = patches.to(device)
        pathway_scores = pathway_scores.to(device)

        out = model(patches, pathway_scores, mode="eval")

        all_cls_p.append(out["cls_p"].cpu())
        all_cls_o.append(out["cls_o"].cpu())
        all_z_wsi.append(out["z_wsi"].cpu())
        all_z_rna.append(out["z_rna"].cpu())

    return {
        "cls_p": torch.cat(all_cls_p, dim=0),   # (N, 256)
        "cls_o": torch.cat(all_cls_o, dim=0),
        "z_wsi": torch.cat(all_z_wsi, dim=0),
        "z_rna": torch.cat(all_z_rna, dim=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 체크포인트 로딩
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path, uni2_dir: str, device: torch.device) -> POMPModel:
    """체크포인트로부터 POMPModel 복원."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = POMPModel(uni2_local_dir=uni2_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    ⚠  missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"    ⚠  unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_results(
    all_results: Dict[str, Dict[str, float]],
    labels: List[str],
    n_patients: int,
):
    """분석 결과를 터미널에 비교 테이블로 출력."""
    col_w  = 20
    name_w = 28
    total_w = name_w + col_w * len(labels)

    print("\n" + "═" * total_w)
    print(f"  POMP_v2 기하학적 분석 결과  (N = {n_patients}명)")
    print("═" * total_w)
    print(f"  {'지표':<{name_w}}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("─" * total_w)

    # (key, 표시이름, 섹션헤더)
    rows = [
        ("cls/cka",                  "CKA ↑",                  "\n  [cls_p / cls_o  ─  projection 이전 CLS]"),
        ("cls/modality_gap",         "Modality Gap ↓",         None),
        ("cls/uniformity_wsi",       "Uniformity WSI ↓",       None),
        ("cls/uniformity_rna",       "Uniformity RNA ↓",       None),
        ("cls/isotropy_wsi",         "Isotropy WSI ↑",         None),
        ("cls/isotropy_rna",         "Isotropy RNA ↑",         None),
        ("cls/cross_modal_isotropy", "Cross-modal Isotropy ↑", None),
        ("z/cka",                    "CKA ↑",                  "\n  [z_wsi / z_rna  ─  L2-normalized ITC]"),
        ("z/modality_gap",           "Modality Gap ↓",         None),
        ("z/uniformity_wsi",         "Uniformity WSI ↓",       None),
        ("z/uniformity_rna",         "Uniformity RNA ↓",       None),
        ("z/isotropy_wsi",           "Isotropy WSI ↑",         None),
        ("z/isotropy_rna",           "Isotropy RNA ↑",         None),
        ("z/cross_modal_isotropy",   "Cross-modal Isotropy ↑", None),
    ]

    for key, name, section in rows:
        if section:
            print(section)
        vals = "".join(
            f"{all_results[label][key]:>{col_w}.4f}"
            if label in all_results and key in all_results[label]
            else f"{'N/A':>{col_w}}"
            for label in labels
        )
        print(f"  {name:<{name_w}}{vals}")

    print("\n" + "─" * total_w)
    print("  ↑: 높을수록 정렬/등방성 good   ↓: 낮을수록 균등/gap 작음 good")
    print("═" * total_w)


def print_interpretation(all_results: Dict[str, Dict[str, float]], labels: List[str]):
    """체크포인트 간 주요 변화 자동 해석."""
    if len(labels) < 2:
        return

    first, best = labels[0], labels[1]
    if first not in all_results or best not in all_results:
        return

    print("\n  [자동 해석: 초기 → Best 변화]")
    for prefix, name in [("cls", "cls_p/cls_o"), ("z", "z_wsi/z_rna")]:
        r0 = all_results[first]
        r1 = all_results[best]

        cka_diff = r1[f"{prefix}/cka"]          - r0[f"{prefix}/cka"]
        gap_diff = r1[f"{prefix}/modality_gap"] - r0[f"{prefix}/modality_gap"]
        iso_diff = r1[f"{prefix}/cross_modal_isotropy"] - r0[f"{prefix}/cross_modal_isotropy"]

        print(f"\n  [{name}]")
        print(f"    CKA:              {r0[f'{prefix}/cka']:.4f} → {r1[f'{prefix}/cka']:.4f}"
              f"  ({'▲' if cka_diff > 0 else '▼'} {abs(cka_diff):.4f})")
        print(f"    Modality Gap:     {r0[f'{prefix}/modality_gap']:.4f} → {r1[f'{prefix}/modality_gap']:.4f}"
              f"  ({'▼' if gap_diff < 0 else '▲'} {abs(gap_diff):.4f})")
        print(f"    Cross-modal Iso:  {r0[f'{prefix}/cross_modal_isotropy']:.4f} → {r1[f'{prefix}/cross_modal_isotropy']:.4f}"
              f"  ({'▲' if iso_diff > 0 else '▼'} {abs(iso_diff):.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="POMP_v2 기하학적 표현 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wsi_dir",   type=str, default="../datasets/preprocessed/wsi",
                   help="전처리된 WSI 디렉토리")
    p.add_argument("--rna_dir",   type=str, default="../datasets/preprocessed/rna",
                   help="전처리된 RNA 디렉토리")
    p.add_argument("--ckpt_dir",  type=str, default="./checkpoints",
                   help="체크포인트 디렉토리")
    p.add_argument("--uni2_dir",  type=str, default="./assets/uni2",
                   help="UNI2-h 모델 로컬 경로")
    p.add_argument("--n_patches", type=int, default=100,
                   help="환자당 샘플링 패치 수")
    p.add_argument("--out",       type=str, default="./geometric_results.json",
                   help="결과 저장 경로 (JSON)")
    p.add_argument("--device",    type=str, default="cuda",
                   help="연산 디바이스")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wsi_dir  = Path(args.wsi_dir)
    rna_dir  = Path(args.rna_dir)
    ckpt_dir = Path(args.ckpt_dir)

    # ── 분석할 체크포인트 정의 ─────────────────────────────────────────────
    checkpoints = [
        ("Epoch 9 (초기)",    ckpt_dir / "epoch_009.pt"),
        ("Epoch 14 (best)",  ckpt_dir / "best.pt"),
        ("Epoch 199 (최종)", ckpt_dir / "epoch_199.pt"),
    ]

    # ── 환자 목록 탐색 ────────────────────────────────────────────────────
    print("\n[Step 1] 환자 목록 탐색 중...")
    patient_ids = find_patients(wsi_dir, rna_dir)

    # ── 패치 인덱스 사전 고정 ─────────────────────────────────────────────
    # 체크포인트 루프 진입 전 1회만 실행.
    # 이후 모든 체크포인트가 동일한 인덱스를 참조하므로
    # RNG 상태나 파일시스템 순서에 무관하게 공정한 비교가 보장됨.
    print("\n[Step 2] 패치 인덱스 사전 고정 중 (seed=42)...")
    patch_indices = precompute_patch_indices(
        patient_ids, wsi_dir, n_patches=args.n_patches, seed=42
    )
    print(f"  완료: {len(patch_indices)}명 × {args.n_patches}패치")

    # ── 체크포인트별 분석 ─────────────────────────────────────────────────
    all_results: Dict[str, Dict[str, float]] = {}
    labels: List[str] = []

    for label, ckpt_path in checkpoints:
        print(f"\n{'─'*60}")
        print(f"[분석] {label}  ({ckpt_path.name})")

        if not ckpt_path.exists():
            print(f"  ⚠  체크포인트 없음, 건너뜀: {ckpt_path}")
            continue

        labels.append(label)

        # 모델 로드
        print("  모델 로딩 중...")
        model = load_model(ckpt_path, args.uni2_dir, device)

        # 임베딩 수집 (고정된 patch_indices 전달)
        print("  임베딩 추출 중...")
        embeds = collect_embeddings(
            model, patient_ids, wsi_dir, rna_dir,
            patch_indices=patch_indices,   # 사전 고정 인덱스
            device=device,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 기하학적 지표 계산
        print("  지표 계산 중...")
        results = compute_all_metrics(
            embeds["cls_p"], embeds["cls_o"],
            embeds["z_wsi"], embeds["z_rna"],
        )
        all_results[label] = results

        # 중간 요약
        print(f"  ┌ CKA (cls / z):            "
              f"{results['cls/cka']:.4f}  /  {results['z/cka']:.4f}")
        print(f"  ├ Modality Gap (cls / z):   "
              f"{results['cls/modality_gap']:.4f}  /  {results['z/modality_gap']:.4f}")
        print(f"  ├ Uniformity WSI (cls / z): "
              f"{results['cls/uniformity_wsi']:.4f}  /  {results['z/uniformity_wsi']:.4f}")
        print(f"  └ Cross-modal Iso (cls / z):"
              f"{results['cls/cross_modal_isotropy']:.4f}  /  {results['z/cross_modal_isotropy']:.4f}")

    # ── 최종 결과 출력 ────────────────────────────────────────────────────
    print_results(all_results, labels, n_patients=len(patient_ids))
    print_interpretation(all_results, labels)

    # ── JSON 저장 ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"n_patients": len(patient_ids), "results": all_results},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n  결과 저장 완료: {out_path}\n")


if __name__ == "__main__":
    main()