"""
RNA 전처리 파이프라인 (PEaRL 방식 + Hallmark)
────────────────────────────────────────────────────────
흐름:
  STAR-Counts TSV 파일
    → TPM 값 추출 (tpm_unstranded 컬럼)
    → log(1 + TPM) 변환
    → ssGSEA (Hallmark gene sets, 50개 pathway)
    → pathway 점수 벡터 저장 (.pt)

의존성:
  pip install gseapy pandas numpy torch tqdm

Hallmark gene set:
  gseapy가 자동으로 MSigDB에서 다운로드.
  또는 수동: https://www.gsea-msigdb.org/gsea/msigdb
  → h.all.v2023.1.Hs.symbols.gmt 다운로드 후 --gmt_path 지정

사용법:
  python rna_preprocessing.py \
      --rna_dir ./downloads/rnaseq \
      --out_dir ./preprocessed/rna

  # gmt 파일 직접 지정 시
  python rna_preprocessing.py \
      --rna_dir ./downloads/rnaseq \
      --out_dir ./preprocessed/rna \
      --gmt_path ./h.all.v2023.1.Hs.symbols.gmt
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

try:
    import gseapy as gp
except ImportError:
    raise ImportError("pip install gseapy 후 실행하세요.")


# ══════════════════════════════════════════════════════════════════════════════
# STAR-Counts 파일 파싱
# ══════════════════════════════════════════════════════════════════════════════
def load_star_counts(tsv_path: str) -> pd.Series:
    """
    GDC STAR-Counts TSV에서 TPM 값 추출.

    GDC STAR-Counts 파일 구조:
        gene_id | gene_name | ... | unstranded | tpm_unstranded | ...
        (앞 4줄: N_unmapped 등 summary 줄 → 스킵)

    Returns:
        pd.Series (index=gene_name, values=tpm)
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#")

    # GDC 파일은 앞 4줄이 통계 요약 → gene_id가 N_으로 시작하는 줄 제거
    if "gene_id" in df.columns:
        df = df[~df["gene_id"].str.startswith("N_", na=False)]
    elif df.columns[0] == df.iloc[0, 0]:
        # 헤더가 없는 경우 처리
        df.columns = ["gene_id", "gene_name", "gene_type",
                      "unstranded", "stranded_first",
                      "stranded_second", "tpm_unstranded",
                      "fpkm_unstranded", "fpkm_uq_unstranded"]
        df = df[~df["gene_id"].str.startswith("N_", na=False)]

    # tpm_unstranded 컬럼 사용
    if "tpm_unstranded" not in df.columns:
        raise ValueError(f"tpm_unstranded 컬럼 없음: {tsv_path}\n"
                         f"실제 컬럼: {list(df.columns)}")

    # gene_name 기준으로 인덱싱 (ssGSEA는 gene symbol 사용)
    if "gene_name" in df.columns:
        df = df.set_index("gene_name")
    else:
        df = df.set_index("gene_id")

    tpm = df["tpm_unstranded"].astype(float)

    # 중복 gene_name 처리: 최대값 유지
    tpm = tpm.groupby(tpm.index).max()

    return tpm


# ══════════════════════════════════════════════════════════════════════════════
# RNA 전처리 클래스
# ══════════════════════════════════════════════════════════════════════════════
class RNAPreprocessor:

    def __init__(self, gmt_path: str = None):
        """
        Args:
            gmt_path: Hallmark gmt 파일 경로.
                      None이면 gseapy가 자동으로 MSigDB에서 다운로드.
        """
        self.gmt_path = gmt_path or "h.all.v2023.1.Hs.symbols.gmt"
        self._gene_sets = None

    def _get_gene_sets(self):
        """Hallmark gene sets 로드 (캐싱)"""
        if self._gene_sets is None:
            if os.path.exists(self.gmt_path):
                self._gene_sets = gp.parser.gsea_gmt_parser(self.gmt_path)
                print(f"[INFO] Hallmark gmt 로드: {self.gmt_path} "
                      f"({len(self._gene_sets)}개 pathway)")
            else:
                # gseapy 내장 데이터셋에서 로드
                print("[INFO] MSigDB에서 Hallmark gene sets 다운로드 중...")
                self._gene_sets = gp.get_library(
                    name="MSigDB_Hallmark_2020",
                    organism="Human"
                )
                print(f"[INFO] Hallmark {len(self._gene_sets)}개 pathway 로드 완료")
        return self._gene_sets

    def _log_transform(self, tpm: pd.Series) -> pd.Series:
        """log(1 + TPM) 변환"""
        return np.log1p(tpm)

    def _run_ssgsea(self, expr: pd.Series) -> pd.Series:
        """
        단일 샘플 ssGSEA 실행.
        Args:
            expr: gene_name → log1p(TPM) Series
        Returns:
            pathway_name → NES score Series (50개 Hallmark)
        """
        gene_sets = self._get_gene_sets()

        # gseapy ssgsea는 DataFrame 입력 (gene × sample)
        expr_df = expr.to_frame(name="sample")

        result = gp.ssgsea(
            data=expr_df,
            gene_sets=gene_sets,
            outdir=None,        # 파일 저장 X
            sample_norm_method="rank",
            no_plot=True,
            processes=1,
            verbose=False,
        )

        # NES score 추출 (pathway × sample)
        nes = result.res2d.pivot(
            index="Term", columns="Name", values="NES"
        )["sample"]

        # ── object → float 강제 변환 ──────────────────────────────────────
        nes = pd.to_numeric(nes, errors="coerce").astype(np.float32)

        return nes

    def process(self, tsv_path: str, out_path: str) -> int:
        """
        단일 환자 RNA 전처리.
        Returns: pathway 수 (50)
        """
        # 1. TPM 로드
        tpm = load_star_counts(tsv_path)

        # 2. log(1+TPM) 변환
        expr = self._log_transform(tpm)

        # 3. ssGSEA → pathway 점수 (50차원)
        nes = self._run_ssgsea(expr)

        # 4. 정렬 (Hallmark 기준 알파벳순 고정)
        nes = nes.sort_index()

        # NaN을 0으로 채우지 말고 에러로 올려서, 어떤 pathway가 문제인지 확인
        if nes.isna().any():
            nan_pathways = nes[nes.isna()].index.tolist()
            raise ValueError(
                f"NaN pathway {len(nan_pathways)}개 발생 — "
                f"유전자 coverage 확인 필요: {nan_pathways[:5]}"
         )

        # 5. 저장
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pathway_tensor = torch.tensor(nes.values, dtype=torch.float32)
        torch.save(pathway_tensor, out_path)

        # pathway 이름도 함께 저장 (첫 번째 환자만)
        pathway_names_path = str(Path(out_path).parent.parent / "pathway_names.txt")
        if not os.path.exists(pathway_names_path):
            with open(pathway_names_path, "w") as f:
                f.write("\n".join(nes.index.tolist()))

        return len(nes)


# ══════════════════════════════════════════════════════════════════════════════
# 배치 처리
# ══════════════════════════════════════════════════════════════════════════════
def run_batch(rna_dir: str, out_dir: str, gmt_path: str = None):
    """
    rna_dir 구조:
        rna_dir/
            TCGA-XX-XXXX/
                *.tsv (STAR-Counts)

    out_dir 구조 (생성):
        out_dir/
            TCGA-XX-XXXX/
                pathway_scores.pt  ← (50,) float32 tensor
            pathway_names.txt      ← 50개 Hallmark pathway 이름
    """
    rna_root   = Path(rna_dir)
    tsv_files  = sorted(rna_root.rglob("*.tsv"))
    print(f"[INFO] TSV 파일 수: {len(tsv_files)}")

    if len(tsv_files) == 0:
        print("[WARN] TSV 파일을 찾을 수 없습니다.")
        return

    preprocessor  = RNAPreprocessor(gmt_path=gmt_path)
    ok = fail = 0
    failed_cases  = []

    for tsv_path in tqdm(tsv_files, desc="RNA ssGSEA"):
        case_id  = tsv_path.parent.name
        out_path = os.path.join(out_dir, case_id, "pathway_scores.pt")

        # 이미 처리된 경우 스킵
        if os.path.exists(out_path):
            ok += 1
            continue

        try:
            n = preprocessor.process(str(tsv_path), out_path)
            print(f"  [OK] {case_id}: {n}개 pathway")
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {case_id}: {e}")
            fail += 1
            failed_cases.append(case_id)

    # 실패 케이스 저장
    if failed_cases:
        fail_path = os.path.join(out_dir, "rna_failed_cases.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(failed_cases))
        print(f"\n[WARN] 실패 케이스 {fail}개 → {fail_path}")

    print(f"\n{'='*50}")
    print(f"  완료: {ok}개 | 실패: {fail}개")
    print(f"  저장 위치: {out_dir}")
    print(f"{'='*50}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RNA 전처리 (ssGSEA, Hallmark)")
    ap.add_argument("--rna_dir",  required=True,
                    help="STAR-Counts TSV 루트 디렉토리")
    ap.add_argument("--out_dir",  required=True,
                    help="전처리 결과 저장 디렉토리")
    ap.add_argument("--gmt_path", default=None,
                    help="Hallmark gmt 파일 경로 (없으면 자동 다운로드)")
    args = ap.parse_args()

    run_batch(
        rna_dir=args.rna_dir,
        out_dir=args.out_dir,
        gmt_path=args.gmt_path,
    )