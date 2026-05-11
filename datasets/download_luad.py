"""
TCGA-LUAD 멀티모달 전체 다운로드 스크립트
- 대상  : TCGA-LUAD 전체 환자 (wsi_files.txt 불필요)
- RNA   : GDC open-access STAR-Counts
- WSI   : GDC SVS 중 파일명에 'DX1' 포함된 슬라이드만
- 실패  : RNA/WSI 중 하나라도 실패한 환자 ID → failed_cases.txt 저장

의존성: pip install requests tqdm

사용법:
  python download_luad.py --dry_run
  python download_luad.py --out_dir ./downloads --rna_only
  python download_luad.py --out_dir ./downloads
"""

import os
import csv
import json
import time
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
PROJECT_ID         = "TCGA-LUAD"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1. TCGA-LUAD 전체 case_id 조회
# ══════════════════════════════════════════════════════════════════════════════
def fetch_all_case_ids(project_id: str = PROJECT_ID) -> list:
    """
    GDC cases 엔드포인트에서 TCGA-LUAD 전체 환자 submitter_id 목록 조회.
    페이지네이션 처리 포함.
    """
    case_ids = []
    page_size = 500
    from_idx  = 0

    filters = {
        "op": "=",
        "content": {
            "field": "project.project_id",
            "value": project_id
        }
    }

    print(f"[STEP 1] {project_id} 전체 환자 조회 중...")
    while True:
        resp = requests.get(GDC_CASES_ENDPOINT, params={
            "filters": json.dumps(filters),
            "fields":  "submitter_id",
            "size":    str(page_size),
            "from":    str(from_idx),
            "format":  "json"
        }, timeout=30)
        resp.raise_for_status()

        data = resp.json()["data"]
        hits = data["hits"]
        if not hits:
            break

        for hit in hits:
            case_ids.append(hit["submitter_id"].upper())

        total = data["pagination"]["total"]
        from_idx += page_size
        print(f"  조회 중: {min(from_idx, total)}/{total}", end="\r")

        if from_idx >= total:
            break
        time.sleep(0.2)

    print(f"\n[STEP 1] 완료: {len(case_ids)}명")
    return case_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2. RNA-seq file_id 조회 (STAR-Counts, open-access)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_rnaseq_uuids(case_ids: list, workflow: str = "STAR - Counts",
                       batch: int = 50) -> dict:
    """
    Returns: {case_id: {"file_id": str, "file_name": str}}
    """
    result = {}
    print(f"\n[STEP 2] RNA-seq 파일 조회 (workflow={workflow})")

    for i in tqdm(range(0, len(case_ids), batch), desc="RNA-seq", unit="batch"):
        b = case_ids[i:i+batch]
        filters = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": b}},
            {"op": "=",  "content": {"field": "data_type",           "value": "Gene Expression Quantification"}},
            {"op": "=",  "content": {"field": "experimental_strategy","value": "RNA-Seq"}},
            {"op": "=",  "content": {"field": "analysis.workflow_type","value": workflow}},
            {"op": "=",  "content": {"field": "access",              "value": "open"}},
        ]}
        resp = requests.get(GDC_FILES_ENDPOINT, params={
            "filters": json.dumps(filters),
            "fields":  "file_id,file_name,cases.submitter_id",
            "size":    str(batch * 3),
            "format":  "json"
        }, timeout=30)
        resp.raise_for_status()

        for hit in resp.json()["data"]["hits"]:
            for case in hit.get("cases", []):
                cid = case["submitter_id"].upper()
                if cid not in result:
                    result[cid] = {
                        "file_id":   hit["file_id"],
                        "file_name": hit["file_name"]
                    }
        time.sleep(0.2)

    print(f"[STEP 2] RNA-seq 매핑: {len(result)}/{len(case_ids)}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3. WSI file_id 조회 (DX1만 필터링)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_wsi_dx1_from_gdc(case_ids: list, batch: int = 50) -> dict:
    """
    GDC에서 SVS 슬라이드 조회 후 파일명에 'DX1' 포함된 것만 선택.
    같은 환자에 DX1이 여러 장이면 첫 번째만 사용 (필요 시 확장 가능).

    Returns: {case_id: {"file_id": str, "file_name": str, "access": str}}
    """
    result     = {}   # DX1 매핑 (case_id → 1개)
    multi_dx1  = {}   # DX1이 2개 이상인 환자 기록용
    print(f"\n[STEP 3] WSI(DX1) 파일 조회")

    for i in tqdm(range(0, len(case_ids), batch), desc="WSI DX1", unit="batch"):
        b = case_ids[i:i+batch]
        filters = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": b}},
            {"op": "=",  "content": {"field": "data_type",   "value": "Slide Image"}},
            {"op": "=",  "content": {"field": "data_format", "value": "SVS"}},
        ]}
        resp = requests.get(GDC_FILES_ENDPOINT, params={
            "filters": json.dumps(filters),
            "fields":  "file_id,file_name,access,cases.submitter_id",
            "size":    str(batch * 10),   # 환자당 슬라이드 여러 장 가능
            "format":  "json"
        }, timeout=30)
        resp.raise_for_status()

        for hit in resp.json()["data"]["hits"]:
            fname = hit["file_name"]

            # ── DX1 필터: 파일명 안에 'DX1' 포함 여부 확인 ──────────────
            # 예) TCGA-XX-XXXX-01Z-00-DX1.xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.svs
            if "DX1" not in fname.upper():
                continue

            for case in hit.get("cases", []):
                cid = case["submitter_id"].upper()
                if cid not in result:
                    result[cid] = {
                        "file_id":   hit["file_id"],
                        "file_name": fname,
                        "access":    hit.get("access", "unknown"),
                    }
                else:
                    # 같은 환자에 DX1이 두 장 이상 → 기록만 해둠
                    multi_dx1.setdefault(cid, [result[cid]["file_name"]])
                    multi_dx1[cid].append(fname)

        time.sleep(0.2)

    open_cnt = sum(1 for v in result.values() if v["access"] == "open")
    ctrl_cnt = sum(1 for v in result.values() if v["access"] == "controlled")
    print(f"[STEP 3] WSI DX1 매핑: {len(result)}/{len(case_ids)} "
          f"(open={open_cnt}, controlled={ctrl_cnt})")

    if multi_dx1:
        print(f"  [주의] DX1 슬라이드 2장 이상 환자: {len(multi_dx1)}명 "
              f"(첫 번째 슬라이드 사용)")
        for cid, fnames in list(multi_dx1.items())[:5]:
            print(f"    {cid}: {fnames}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4. 매핑 테이블 빌드 & CSV 저장
# ══════════════════════════════════════════════════════════════════════════════
def build_and_save_mapping(case_ids: list, rnaseq_map: dict,
                           wsi_map: dict, out_path: str) -> list:
    rows = []
    for cid in case_ids:
        rna = rnaseq_map.get(cid)
        wsi = wsi_map.get(cid)

        # controlled WSI는 다운로드 불가이므로 paired 불가 처리
        wsi_available = wsi and wsi["access"] == "open"

        row = {
            "case_id":       cid,
            "rna_file_id":   rna["file_id"]   if rna else "",
            "rna_file_name": rna["file_name"] if rna else "",
            "wsi_file_id":   wsi["file_id"]   if wsi else "",
            "wsi_file_name": wsi["file_name"] if wsi else "",
            "wsi_access":    wsi["access"]    if wsi else "",
            "paired":        str(bool(rna and wsi_available)),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["case_id", "rna_file_id", "rna_file_name",
              "wsi_file_id", "wsi_file_name", "wsi_access", "paired"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    paired      = sum(1 for r in rows if r["paired"] == "True")
    rna_only    = sum(1 for r in rows if r["rna_file_id"] and not r["wsi_file_id"])
    wsi_only    = sum(1 for r in rows if r["wsi_file_id"] and not r["rna_file_id"])
    no_data     = sum(1 for r in rows if not r["rna_file_id"] and not r["wsi_file_id"])
    controlled  = sum(1 for r in rows if r["wsi_access"] == "controlled")

    print(f"\n{'='*60}")
    print(f"  전체 환자          : {len(rows)}")
    print(f"  RNA-seq 있음       : {sum(1 for r in rows if r['rna_file_id'])}")
    print(f"  WSI DX1 있음       : {sum(1 for r in rows if r['wsi_file_id'])}")
    print(f"    └ controlled(불가): {controlled}")
    print(f"  페어링 가능        : {paired}")
    print(f"  RNA만 있음         : {rna_only}")
    print(f"  WSI만 있음         : {wsi_only}")
    print(f"  둘 다 없음         : {no_data}")
    print(f"{'='*60}")
    print(f"[STEP 4] 매핑 저장: {out_path}\n")

    print(f"{'case_id':<20} {'wsi_file_name':<45} {'paired'}")
    print("-"*75)
    for r in rows[:10]:
        print(f"{r['case_id']:<20} {r['wsi_file_name'][:43]:<45} {r['paired']}")
    if len(rows) > 10:
        print(f"  ... ({len(rows)-10}개 더)")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5. 파일 다운로드 (스트리밍)
# ══════════════════════════════════════════════════════════════════════════════
def download_file(url: str, save_path: str) -> bool:
    """
    이미 파일이 있으면 스킵. 실패 시 부분 파일 삭제.
    Returns True if success (including already-exists).
    """
    if os.path.exists(save_path):
        return True
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 401:
            print(f"  [401 SKIP] controlled: {os.path.basename(save_path)}")
            return False
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=os.path.basename(save_path)[:45], leave=False
        ) as bar:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(save_path)}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6. 실패 케이스 저장
# ══════════════════════════════════════════════════════════════════════════════
def save_failed_cases(failed: dict, out_path: str):
    """
    failed: {case_id: {"rna": bool, "wsi": bool, "reason": str}}
    """
    if not failed:
        print("[INFO] 실패 케이스 없음.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "rna_ok", "wsi_ok", "reason"])
        w.writeheader()
        for cid, info in failed.items():
            w.writerow({
                "case_id": cid,
                "rna_ok":  str(info["rna_ok"]),
                "wsi_ok":  str(info["wsi_ok"]),
                "reason":  info["reason"],
            })
    print(f"[STEP 6] 실패 케이스 {len(failed)}명 저장: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: 전체 환자 목록 ──────────────────────────────────────────────
    case_ids = fetch_all_case_ids(PROJECT_ID)

    # ── STEP 2: RNA-seq 조회 ────────────────────────────────────────────────
    rnaseq_map = fetch_rnaseq_uuids(case_ids, workflow=args.workflow)

    # ── STEP 3: WSI DX1 조회 ────────────────────────────────────────────────
    wsi_map = fetch_wsi_dx1_from_gdc(case_ids)

    # ── STEP 4: 매핑 테이블 ─────────────────────────────────────────────────
    mapping_path = str(out_dir / "mapping.csv")
    rows = build_and_save_mapping(case_ids, rnaseq_map, wsi_map, mapping_path)

    if args.dry_run:
        print("[DRY RUN] 다운로드 없이 종료.")
        return

    # ── 페어링된 케이스만 다운로드 대상 ─────────────────────────────────────
    paired_rows = [r for r in rows if r["paired"] == "True"]
    print(f"\n페어링 완료 케이스: {len(paired_rows)}명 다운로드 시작\n")

    # 실패 추적: {case_id: {rna_ok, wsi_ok, reason}}
    failed = {}

    # ── STEP 5-A: RNA-seq 다운로드 ──────────────────────────────────────────
    if not args.wsi_only:
        print(f"=== RNA-seq 다운로드 ({len(paired_rows)}개) ===")
        for r in tqdm(paired_rows, desc="RNA-seq", unit="case"):
            cid  = r["case_id"]
            url  = f"{GDC_DATA_ENDPOINT}/{r['rna_file_id']}"
            path = str(out_dir / "rnaseq" / cid / r["rna_file_name"])
            ok   = download_file(url, path)
            if not ok:
                failed.setdefault(cid, {"rna_ok": True, "wsi_ok": True, "reason": ""})
                failed[cid]["rna_ok"] = False
                failed[cid]["reason"] += "RNA 다운로드 실패; "

    # ── STEP 5-B: WSI 다운로드 ──────────────────────────────────────────────
    if not args.rna_only:
        print(f"\n=== WSI DX1 다운로드 ({len(paired_rows)}개) ===")
        for r in tqdm(paired_rows, desc="WSI", unit="case"):
            cid  = r["case_id"]
            url  = f"{GDC_DATA_ENDPOINT}/{r['wsi_file_id']}"
            path = str(out_dir / "wsi" / cid / r["wsi_file_name"])
            ok   = download_file(url, path)
            if not ok:
                failed.setdefault(cid, {"rna_ok": True, "wsi_ok": True, "reason": ""})
                failed[cid]["wsi_ok"] = False
                failed[cid]["reason"] += "WSI 다운로드 실패; "

    # ── STEP 6: 실패 케이스 저장 ────────────────────────────────────────────
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_path  = str(out_dir / f"failed_cases_{ts}.csv")
    save_failed_cases(failed, failed_path)

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    total    = len(paired_rows)
    n_failed = len(failed)
    print(f"\n{'='*60}")
    print(f"  다운로드 대상 : {total}명")
    print(f"  성공          : {total - n_failed}명")
    print(f"  실패          : {n_failed}명  →  {failed_path}")
    print(f"  저장 위치     : {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TCGA-LUAD 전체 환자 RNA-seq + WSI(DX1) 다운로드"
    )
    ap.add_argument("--out_dir",  default="./downloads",
                    help="다운로드 저장 경로 (기본: ./downloads)")
    ap.add_argument("--workflow", default="STAR - Counts",
                    choices=["STAR - Counts", "STAR - FPKM", "STAR - FPKM-UQ"],
                    help="RNA-seq 워크플로우 타입")
    ap.add_argument("--dry_run",  action="store_true",
                    help="매핑 테이블만 생성하고 다운로드 안 함")
    ap.add_argument("--rna_only", action="store_true",
                    help="RNA-seq만 다운로드")
    ap.add_argument("--wsi_only", action="store_true",
                    help="WSI만 다운로드")
    args = ap.parse_args()
    main(args)