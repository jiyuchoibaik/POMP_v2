"""
WSI 전처리 파이프라인 (POMP 방식)
────────────────────────────────────────────────────────
흐름:
  SVS 파일
    → HSV 조직 마스킹 (Otsu + Morphological opening)
    → 4096×4096 non-overlapping 패치 추출 (20x)
    → 조직 비율 < 50% 패치 제거
    → 256×256 다운샘플링
    → Macenko 색상 정규화
    → .pt 파일로 저장 (환자당 폴더)

의존성:
  pip install openslide-python opencv-python numpy torch torchvision tqdm
  (Ubuntu: sudo apt-get install openslide-tools)

사용법:
  python wsi_preprocessing.py \
      --wsi_dir ./downloads/wsi \
      --out_dir ./preprocessed/wsi \
      --num_workers 4
"""

import os
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import openslide
except ImportError:
    raise ImportError("pip install openslide-python 후 실행하세요.")


# ══════════════════════════════════════════════════════════════════════════════
# Macenko 색상 정규화
# ══════════════════════════════════════════════════════════════════════════════
class MacenkoNormalizer:
    """
    Macenko et al. (2009) H&E 색상 정규화.
    Reference stain matrix는 TCGA 표준 reference 이미지 기반.
    """

    def __init__(self):
        # TCGA 표준 reference stain matrix (H, E 순서)
        self.HE_ref = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.max_C_ref = np.array([1.9705, 1.0308])

    def _rgb2od(self, img: np.ndarray) -> np.ndarray:
        """RGB → Optical Density 변환"""
        img = img.astype(np.float64) + 1e-6
        return -np.log(img / 255.0)

    def _od2rgb(self, od: np.ndarray) -> np.ndarray:
        """Optical Density → RGB 변환"""
        rgb = np.exp(-od) * 255
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def normalize(self, img: np.ndarray,
                  beta: float = 0.15,
                  alpha: float = 1.0) -> np.ndarray:
        """
        Args:
            img: H×W×3 uint8 RGB 이미지
            beta: 배경 제거 OD 임계값
            alpha: percentile 범위 (1~99)
        Returns:
            정규화된 H×W×3 uint8 RGB 이미지
        """
        h, w = img.shape[:2]
        od = self._rgb2od(img).reshape(-1, 3)

        # 배경 제거 (OD norm이 beta 미만인 픽셀 제거)
        od_norm = np.linalg.norm(od, axis=1)
        tissue_mask = od_norm > beta
        if tissue_mask.sum() < 10:
            return img  # 조직이 너무 적으면 원본 반환

        od_tissue = od[tissue_mask]

        # SVD로 stain 방향 추출
        _, _, V = np.linalg.svd(od_tissue, full_matrices=False)
        V = V[:2]  # 상위 2개 성분만 사용

        # 각도 기반으로 HE stain 벡터 결정
        that = od_tissue @ V.T
        phi = np.arctan2(that[:, 1], that[:, 0])

        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)

        v1 = V.T @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = V.T @ np.array([np.cos(max_phi), np.sin(max_phi)])

        # H 채널이 더 강한 쪽을 Hematoxylin으로 지정
        HE = np.array([v1, v2]).T
        if HE[0, 0] < HE[0, 1]:
            HE = HE[:, [1, 0]]

        # stain concentration 계산
        C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
        max_C = np.percentile(C, 99, axis=1)

        # Reference로 정규화
        C_norm = C * (self.max_C_ref[:, None] / (max_C[:, None] + 1e-6))

        # OD 재구성 → RGB 변환
        od_norm_img = self.HE_ref @ C_norm
        img_norm = self._od2rgb(od_norm_img.T.reshape(h, w, 3))

        return img_norm


# ══════════════════════════════════════════════════════════════════════════════
# WSI 전처리 클래스
# ══════════════════════════════════════════════════════════════════════════════
class WSIPreprocessor:

    def __init__(
        self,
        patch_size: int   = 4096,   # 추출 패치 크기 (원본 해상도 기준)
        output_size: int  = 256,    # 다운샘플 후 크기
        tissue_thresh: float = 0.5, # 조직 비율 임계값
        target_mag: float = 20.0,   # 목표 배율
    ):
        self.patch_size    = patch_size
        self.output_size   = output_size
        self.tissue_thresh = tissue_thresh
        self.target_mag    = target_mag
        self.normalizer    = MacenkoNormalizer()

    # ── 배율 레벨 탐색 ────────────────────────────────────────────────────
    def _get_target_level(self, slide: "openslide.OpenSlide") -> int:
        """20x에 해당하는 OpenSlide 레벨 반환"""
        objective = float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        downsample_target = objective / self.target_mag

        best_level = 0
        best_diff  = float("inf")
        for lvl, ds in enumerate(slide.level_downsamples):
            diff = abs(ds - downsample_target)
            if diff < best_diff:
                best_diff  = diff
                best_level = lvl
        return best_level

    # ── 조직 마스크 생성 ─────────────────────────────────────────────────
    def _get_tissue_mask(self, slide: "openslide.OpenSlide") -> np.ndarray:
        """
        저해상도 썸네일로 조직 마스크 생성.
        HSV S채널 → Otsu thresholding → Morphological opening
        Returns: (H, W) bool mask (thumbnail 해상도)
        """
        # 썸네일 크기로 읽기 (긴 변 기준 ~2000px)
        thumb_size = (2000, 2000)
        thumb = slide.get_thumbnail(thumb_size)
        thumb = np.array(thumb.convert("RGB"))

        # HSV 변환 후 Saturation 채널 사용
        hsv   = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
        s_ch  = hsv[:, :, 1]

        # Otsu thresholding
        _, binary = cv2.threshold(
            s_ch, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological opening (노이즈 제거)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return mask.astype(bool), thumb.shape[:2]

    # ── 조직 비율 계산 ────────────────────────────────────────────────────
    def _tissue_ratio(self, mask_region: np.ndarray) -> float:
        """해당 패치 영역의 조직 비율"""
        if mask_region.size == 0:
            return 0.0
        return mask_region.sum() / mask_region.size

    # ── 단일 패치 처리 ────────────────────────────────────────────────────
    def _process_patch(
        self,
        slide:     "openslide.OpenSlide",
        level:     int,
        x:         int,   # 원본 해상도 x 좌표
        y:         int,   # 원본 해상도 y 좌표
        mask:      np.ndarray,
        mask_h:    int,
        mask_w:    int,
        slide_w:   int,
        slide_h:   int,
    ):
        """
        단일 4096×4096 패치 추출 → 조직 필터 → 256×256 다운샘플 → Macenko 정규화
        Returns: (np.ndarray H×W×3 uint8) or None
        """
        # 마스크 좌표 계산
        mx = int(x / slide_w * mask_w)
        my = int(y / slide_h * mask_h)
        mw = int(self.patch_size / slide_w * mask_w)
        mh = int(self.patch_size / slide_h * mask_h)

        mask_patch = mask[
            my: min(my + mh, mask_h),
            mx: min(mx + mw, mask_w)
        ]

        # 조직 비율 < 50% → 제거
        if self._tissue_ratio(mask_patch) < self.tissue_thresh:
            return None

        # 패치 읽기 (level 해상도 기준 크기 조정)
        level_ds   = slide.level_downsamples[level]
        read_size  = int(self.patch_size / level_ds)

        region = slide.read_region(
            location=(x, y),
            level=level,
            size=(read_size, read_size)
        )
        patch = np.array(region.convert("RGB"))

        # 256×256 다운샘플링
        patch_small = cv2.resize(
            patch,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA
        )

        # Macenko 색상 정규화
        try:
            patch_norm = self.normalizer.normalize(patch_small)
        except Exception:
            patch_norm = patch_small  # 정규화 실패 시 원본 사용

        return patch_norm

    # ── 전체 WSI 처리 ─────────────────────────────────────────────────────
    def process(self, svs_path: str, out_dir: str) -> int:
        """
        WSI 1개 전처리.
        Returns: 저장된 패치 수
        """
        slide = openslide.OpenSlide(svs_path)
    
    # mag=40x만 처리, 나머지는 -1 반환
        mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if mag is None or float(mag) != 40.0:
            print(f"  [SKIP] mag={mag}: {Path(svs_path).name}")
            slide.close()
            return -1

        level    = self._get_target_level(slide)
        slide_w, slide_h = slide.dimensions
        mask, (mask_h, mask_w) = self._get_tissue_mask(slide)

        patches = []
        # 4096×4096 grid (원본 해상도 기준)
        for y in range(0, slide_h - self.patch_size + 1, self.patch_size):
            for x in range(0, slide_w - self.patch_size + 1, self.patch_size):
                patch = self._process_patch(
                    slide, level, x, y,
                    mask, mask_h, mask_w,
                    slide_w, slide_h
                )
                if patch is not None:
                    patches.append(patch)

        slide.close()

        if len(patches) == 0:
            print(f"  [WARN] 패치 없음: {Path(svs_path).name}")
            return 0

        # (N, 256, 256, 3) → (N, 3, 256, 256) tensor
        patches_arr = np.stack(patches)                    # (N, 256, 256, 3)
        patches_t   = torch.from_numpy(
            patches_arr.transpose(0, 3, 1, 2)             # (N, 3, 256, 256)
        ).float() / 255.0

        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "patches.pt")
        torch.save(patches_t, save_path)

        return len(patches)


# ══════════════════════════════════════════════════════════════════════════════
# 배치 처리
# ══════════════════════════════════════════════════════════════════════════════
def process_one(args):
    """멀티프로세싱용 래퍼"""
    svs_path, out_dir, patch_size, output_size, tissue_thresh = args
    preprocessor = WSIPreprocessor(
        patch_size=patch_size,
        output_size=output_size,
        tissue_thresh=tissue_thresh,
    )
    try:
        n = preprocessor.process(svs_path, out_dir)
        return svs_path, n, None
    except Exception as e:
        return svs_path, 0, str(e)


def run_batch(wsi_dir: str, out_dir: str,
              patch_size: int = 4096,
              output_size: int = 256,
              tissue_thresh: float = 0.5,
              num_workers: int = 1):
    wsi_root = Path(wsi_dir)
    svs_files = sorted(wsi_root.rglob("*.svs"))
    print(f"[INFO] SVS 파일 수: {len(svs_files)}")

    if len(svs_files) == 0:
        print("[WARN] SVS 파일을 찾을 수 없습니다.")
        return

    tasks = []
    for svs_path in svs_files:
        case_id   = svs_path.parent.name
        case_out  = os.path.join(out_dir, case_id)
        if os.path.exists(os.path.join(case_out, "patches.pt")):
            continue
        tasks.append((str(svs_path), case_out,
                      patch_size, output_size, tissue_thresh))

    print(f"[INFO] 처리 대상: {len(tasks)}개 (이미 완료: {len(svs_files)-len(tasks)}개 스킵)")

    ok = fail = skip = 0          # ← skip 추가
    failed_cases  = []
    skipped_cases = []            # ← 추가

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="WSI 전처리"):
                svs_path, n_patches, err = fut.result()
                case_id = Path(svs_path).parent.name
                if err:
                    print(f"  [ERROR] {case_id}: {err}")
                    fail += 1
                    failed_cases.append(case_id)
                elif n_patches == -1:                     # ← 추가
                    skip += 1
                    skipped_cases.append(case_id)
                else:
                    print(f"  [OK] {case_id}: {n_patches}개 패치")
                    ok += 1
    else:
        for t in tqdm(tasks, desc="WSI 전처리"):
            svs_path, n_patches, err = process_one(t)
            case_id = Path(svs_path).parent.name
            if err:
                print(f"  [ERROR] {case_id}: {err}")
                fail += 1
                failed_cases.append(case_id)
            elif n_patches == -1:                         # ← 추가
                skip += 1
                skipped_cases.append(case_id)
            else:
                print(f"  [OK] {case_id}: {n_patches}개 패치")
                ok += 1

    if failed_cases:
        fail_path = os.path.join(out_dir, "wsi_failed_cases.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(failed_cases))
        print(f"\n[WARN] 실패 케이스 {fail}개 → {fail_path}")

    if skipped_cases:                                     # ← 추가
        skip_path = os.path.join(out_dir, "wsi_skipped_cases.txt")
        with open(skip_path, "w") as f:
            f.write("\n".join(skipped_cases))
        print(f"[INFO] 스킵 케이스 {skip}개 → {skip_path}")

    print(f"\n{'='*50}")
    print(f"  완료: {ok}개 | 스킵: {skip}개 | 실패: {fail}개")  # ← skip 추가
    print(f"  저장 위치: {out_dir}")
    print(f"{'='*50}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WSI 전처리 (POMP 방식)")
    ap.add_argument("--wsi_dir",       required=True,
                    help="SVS 파일 루트 디렉토리")
    ap.add_argument("--out_dir",       required=True,
                    help="전처리 결과 저장 디렉토리")
    ap.add_argument("--patch_size",    type=int, default=4096,
                    help="원본 해상도 기준 패치 크기 (default: 4096)")
    ap.add_argument("--output_size",   type=int, default=256,
                    help="다운샘플 후 크기 (default: 256)")
    ap.add_argument("--tissue_thresh", type=float, default=0.5,
                    help="조직 비율 임계값 (default: 0.5)")
    ap.add_argument("--num_workers",   type=int, default=1,
                    help="병렬 처리 worker 수 (default: 1)")
    args = ap.parse_args()

    run_batch(
        wsi_dir=args.wsi_dir,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        output_size=args.output_size,
        tissue_thresh=args.tissue_thresh,
        num_workers=args.num_workers,
    )