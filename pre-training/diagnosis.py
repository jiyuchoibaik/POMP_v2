# 진단 스크립트 - diagnosis.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_loader import POMPDataset, pomp_collate_fn
from pomp import POMPModel

device = torch.device("cuda")

# best checkpoint 로드
ckpt  = torch.load("./checkpoints/best.pt", map_location="cpu")
model = POMPModel(
    uni2_local_dir="./assets/uni2",
    total_epochs=200,
    queue_size=256,
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# 데이터 로드
ds = POMPDataset(
    wsi_dir="../datasets/preprocessed/wsi",
    rna_dir="../datasets/preprocessed/rna",
    n_patches=100,
    split="all",
)
loader = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=pomp_collate_fn, num_workers=2)

# z_wsi, z_rna 전체 수집
all_z_wsi, all_z_rna, all_case_ids = [], [], []
with torch.no_grad():
    for batch in loader:
        patches        = batch["patches"].to(device)
        pathway_scores = batch["pathway_scores"].to(device)

        cls_p, wsi_tokens = model.wsi_encoder(patches)
        cls_o, rna_tokens = model.rna_encoder(pathway_scores)
        out_mm = model.mm_encoder(wsi_tokens, rna_tokens, cls_p, cls_o)

        all_z_wsi.append(out_mm["z_wsi"].cpu())
        all_z_rna.append(out_mm["z_rna"].cpu())
        all_case_ids.extend(batch["case_ids"])

z_wsi = torch.cat(all_z_wsi)   # (N, 256)
z_rna = torch.cat(all_z_rna)   # (N, 256)

# 1. Modality Gap 확인
gap = (z_wsi.mean(0) - z_rna.mean(0)).norm().item()
print(f"Modality Gap: {gap:.4f}")

# 2. 같은 환자 cosine similarity (대각선)
sim = z_wsi @ z_rna.T   # (N, N)
pos_sim = sim.diag().mean().item()
print(f"Positive pair similarity (mean): {pos_sim:.4f}")

# 3. 다른 환자 cosine similarity (비대각선)
mask = ~torch.eye(len(z_wsi), dtype=torch.bool)
neg_sim = sim[mask].mean().item()
print(f"Negative pair similarity (mean): {neg_sim:.4f}")

# 4. z_wsi, z_rna 분포 확인
print(f"\nz_wsi norm: {z_wsi.norm(dim=-1).mean():.4f} (정규화 후 1이어야 함)")
print(f"z_rna norm: {z_rna.norm(dim=-1).mean():.4f}")

# 5. Rank@1 정확도
ranks = (sim.argsort(dim=-1, descending=True) == 
         torch.arange(len(z_wsi)).unsqueeze(1)).nonzero()[:, 1]
r1 = (ranks == 0).float().mean().item()
print(f"\nRecall@1 (WSI→RNA): {r1:.4f}  (random={1/len(z_wsi):.4f})")