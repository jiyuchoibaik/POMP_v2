# 전처리 완료 후 패치 수 분포 확인용
import os
import torch
from pathlib import Path

preprocessed_dir = "./preprocessed/wsi"
counts = {}

for case_dir in Path(preprocessed_dir).iterdir():
    pt_path = case_dir / "patches.pt"
    if pt_path.exists():
        patches = torch.load(pt_path, weights_only=True)
        counts[case_dir.name] = patches.shape[0]

values = list(counts.values())
print(f"총 환자 수  : {len(values)}")
print(f"평균 패치 수: {sum(values)/len(values):.1f}")
print(f"중앙값      : {sorted(values)[len(values)//2]}")
print(f"최솟값      : {min(values)}")
print(f"최댓값      : {max(values)}")
print(f"10개 미만   : {sum(1 for v in values if v < 10)}명")
print(f"20개 미만   : {sum(1 for v in values if v < 20)}명")
print(f"30개 미만   : {sum(1 for v in values if v < 30)}명")

# 분포 히스토그램
import collections
buckets = collections.Counter(v // 50 * 50 for v in values)
for k in sorted(buckets):
    print(f"  {k:>4}~{k+49}: {'█' * buckets[k]} ({buckets[k]}명)")