# POMP_v2/pretraining/pomp.py
"""
POMP 전체 모델
────────────────────────────────────────────────────────
수정 사항:
  [1] Loss weight: 1:2:1
  [2] Hard negative curriculum (초기: random, 후반: hard)
  [3] eval return 확장 (cls_p, cls_o, z_wsi, z_rna, fusion_wsi, fusion_rna)
  [4] ITC projection을 pomp.py로 분리 (mm_encoder에서 제거)
  [5] Memory Queue (MoCo 방식)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from wsi_encoder import WSIEncoder
from rna_encoder import RNAEncoder
from multimodal_encoder import MultimodalEncoder


class POMPModel(nn.Module):
    def __init__(
        self,
        # WSI 인코더
        uni2_local_dir: str   = "./assets/uni2",
        pat_hidden_dim: int   = 512,
        pat_output_dim: int   = 256,
        pat_depth: int        = 2,
        pat_num_heads: int    = 8,
        # RNA 인코더
        num_pathways: int     = 50,
        rna_hidden_dim: int   = 512,
        rna_output_dim: int   = 256,
        rna_depth: int        = 2,
        rna_num_heads: int    = 8,
        # 멀티모달 인코더
        mm_dim: int           = 512,
        mm_depth: int         = 2,
        mm_num_heads: int     = 8,
        # 공통
        drop_rate: float      = 0.1,
        attn_drop_rate: float = 0.1,
        mask_ratio: float     = 0.15,
        queue_size: int       = 256,
        # loss 가중치
        itc_weight: float     = 1.0,
        itm_weight: float     = 2.0,
        mom_weight: float     = 1.0,
        # ITC temperature
        temp: float           = 0.07,
        # hard negative curriculum
        total_epochs: int     = 100,
        hard_negative_start_epoch: int = None,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.mom_weight = mom_weight
        self.queue_size = queue_size

        # ── ITC projection (mm_encoder와 분리) ────────────────────────────
        self.wsi_proj = nn.Sequential(
            nn.Linear(pat_output_dim, pat_output_dim),
            nn.GELU(),
            nn.Linear(pat_output_dim, pat_output_dim),
        )
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_output_dim, rna_output_dim),
            nn.GELU(),
            nn.Linear(rna_output_dim, rna_output_dim),
        )

        # ── Memory Queue ──────────────────────────────────────────────────
        self.register_buffer("queue_wsi",
            F.normalize(torch.randn(queue_size, pat_output_dim), dim=-1))
        self.register_buffer("queue_rna",
            F.normalize(torch.randn(queue_size, rna_output_dim), dim=-1))
        self.register_buffer("queue_ptr",
            torch.zeros(1, dtype=torch.long))

        # ── Hard negative curriculum ──────────────────────────────────────
        self.hard_negative_start_epoch = (
            hard_negative_start_epoch
            if hard_negative_start_epoch is not None
            else int(total_epochs * 0.3)
        )
        self.itm_start_epoch = int(total_epochs * 0.2)
        self.current_epoch   = 0

        # ── Learnable temperature ─────────────────────────────────────────
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1 / temp)
        )

        # ── 단일 모달 인코더 ──────────────────────────────────────────────
        self.wsi_encoder = WSIEncoder(
            uni2_local_dir=uni2_local_dir,
            pat_hidden_dim=pat_hidden_dim,
            pat_output_dim=pat_output_dim,
            pat_depth=pat_depth,
            pat_num_heads=pat_num_heads,
            drop_rate=drop_rate,
        )
        self.rna_encoder = RNAEncoder(
            num_pathways=num_pathways,
            hidden_dim=rna_hidden_dim,
            output_dim=rna_output_dim,
            depth=rna_depth,
            num_heads=rna_num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        # ── 멀티모달 인코더 ───────────────────────────────────────────────
        self.mm_encoder = MultimodalEncoder(
            dim=mm_dim,
            cls_dim=pat_output_dim,
            num_heads=mm_num_heads,
            depth=mm_depth,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            num_pathways=num_pathways,
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    # ── Memory Queue 업데이트 ─────────────────────────────────────────────
    @torch.no_grad()
    def _dequeue_and_enqueue(self, z_wsi: torch.Tensor, z_rna: torch.Tensor):
        B   = z_wsi.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + B > self.queue_size:
            remain = self.queue_size - ptr
            self.queue_wsi[ptr:]      = z_wsi[:remain].detach()
            self.queue_rna[ptr:]      = z_rna[:remain].detach()
            self.queue_wsi[:B-remain] = z_wsi[remain:].detach()
            self.queue_rna[:B-remain] = z_rna[remain:].detach()
            ptr = B - remain
        else:
            self.queue_wsi[ptr:ptr+B] = z_wsi.detach()
            self.queue_rna[ptr:ptr+B] = z_rna.detach()
            ptr = (ptr + B) % self.queue_size

        self.queue_ptr[0] = ptr

    # ── ITC loss ──────────────────────────────────────────────────────────
    def _itc_loss(self, z_wsi: torch.Tensor, z_rna: torch.Tensor) -> torch.Tensor:
        B = z_wsi.shape[0]
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # queue 없이 in-batch negatives만 사용 (SimCLR 방식)
        sim_w2r = logit_scale * z_wsi @ z_rna.T   # (B, B+Q)
        sim_r2w = logit_scale * z_rna @ z_wsi.T   # (B, B+Q)

        labels = torch.arange(B, device=z_wsi.device)

        loss_w2r = F.cross_entropy(sim_w2r, labels)
        loss_r2w = F.cross_entropy(sim_r2w, labels)

        return (loss_w2r + loss_r2w) / 2

    # ── Negative 샘플링 (curriculum) ──────────────────────────────────────
    def _sample_negatives(
        self, z_wsi: torch.Tensor, z_rna: torch.Tensor
    ) -> tuple:
        B      = z_wsi.shape[0]
        device = z_wsi.device

        if self.current_epoch < self.hard_negative_start_epoch:
            neg_rna_idx = torch.zeros(B, dtype=torch.long, device=device)
            neg_wsi_idx = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                candidates = list(range(B))
                candidates.remove(b)
                neg_rna_idx[b] = candidates[torch.randint(len(candidates), (1,)).item()]
                neg_wsi_idx[b] = candidates[torch.randint(len(candidates), (1,)).item()]
        else:
            with torch.no_grad():
                logit_scale = self.logit_scale.exp().clamp(max=100)
                sim_w2r = logit_scale * z_wsi @ z_rna.T

                weights_w2r = sim_w2r.clone().fill_diagonal_(float("-inf")).softmax(dim=-1)
                weights_r2w = sim_w2r.T.clone().fill_diagonal_(float("-inf")).softmax(dim=-1)

                neg_rna_idx = torch.multinomial(weights_w2r, 1).squeeze(1)
                neg_wsi_idx = torch.multinomial(weights_r2w, 1).squeeze(1)

        return neg_rna_idx, neg_wsi_idx

    # ── ITM + MOM loss ────────────────────────────────────────────────────
    def _itm_mom_loss(
        self,
        wsi_tokens:    torch.Tensor,
        rna_tokens:    torch.Tensor,
        z_wsi:         torch.Tensor,
        z_rna:         torch.Tensor,
        target_scores: torch.Tensor,
        rna_mask:      torch.Tensor,
    ) -> tuple:
        B      = wsi_tokens.shape[0]
        device = wsi_tokens.device

        neg_rna_idx, _ = self._sample_negatives(z_wsi, z_rna)

        # Positive pair (MOM 포함)
        out_pos = self.mm_encoder(
            wsi_tokens, rna_tokens,
            rna_mask=rna_mask,
            target_scores=target_scores,
        )

        # Negative pair: WSI 고정, RNA negative (1:1 비율)
        out_neg = self.mm_encoder(
            wsi_tokens, rna_tokens[neg_rna_idx],
        )

        # ITM loss (1:1 positive:negative)
        itm_logits = torch.cat([
            out_pos["itm_logits"],
            out_neg["itm_logits"],
        ], dim=0)                                    # (2B, 2)

        itm_labels = torch.cat([
            torch.ones(B,  dtype=torch.long),
            torch.zeros(B, dtype=torch.long),
        ], dim=0).to(device)

        loss_itm = F.cross_entropy(itm_logits, itm_labels, label_smoothing=0.1)
        loss_mom = out_pos["loss_mom"]

        fusion_wsi = out_pos["fusion_wsi"]
        fusion_rna = out_pos["fusion_rna"]

        return loss_itm, loss_mom, fusion_wsi, fusion_rna

    # ── MOM 마스킹 생성 ───────────────────────────────────────────────────
    def _make_rna_mask(self, B: int, device: torch.device) -> torch.Tensor:
        num_mask = max(1, int(self.mask_ratio * 50))
        mask = torch.zeros(B, 50, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(50, device=device)[:num_mask]
            mask[b, idx] = True
        return mask

    # ── forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        patches:        torch.Tensor,
        pathway_scores: torch.Tensor,
        mode:           str = "pretrain",
    ) -> dict:
        B      = patches.shape[0]
        device = patches.device

        # 단일 모달 인코더
        cls_p, wsi_tokens = self.wsi_encoder(patches)
        cls_o, rna_tokens = self.rna_encoder(pathway_scores)

        # ITC projection (mm_encoder와 독립)
        z_wsi = F.normalize(self.wsi_proj(cls_p), dim=-1)  # (B, 256)
        z_rna = F.normalize(self.rna_proj(cls_o), dim=-1)  # (B, 256)

        if mode == "eval":
            out_mm = self.mm_encoder(wsi_tokens, rna_tokens)
            return {
                "cls_p":      cls_p,
                "cls_o":      cls_o,
                "z_wsi":      z_wsi,
                "z_rna":      z_rna,
                "fusion_wsi": out_mm["fusion_wsi"],
                "fusion_rna": out_mm["fusion_rna"],
            }

        rna_mask = self._make_rna_mask(B, device)

        loss_itc = self._itc_loss(z_wsi, z_rna)

        loss_itm, loss_mom, fusion_wsi, fusion_rna = self._itm_mom_loss(
            wsi_tokens, rna_tokens,
            z_wsi, z_rna,
            target_scores=pathway_scores,
            rna_mask=rna_mask,
        )

        itm_weight = self.itm_weight if self.current_epoch >= self.itm_start_epoch else 0.0

        loss = (
            self.itc_weight * loss_itc +
            itm_weight      * loss_itm +
            self.mom_weight * loss_mom
        )

        return {
            "loss":       loss,
            "loss_itc":   loss_itc,
            "loss_itm":   loss_itm,
            "loss_mom":   loss_mom,
            "itm_active": self.current_epoch >= self.itm_start_epoch,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    B, N = 2, 50
    patches        = torch.rand(B, N, 3, 256, 256)
    pathway_scores = torch.rand(B, 50)

    model = POMPModel(total_epochs=100)

    model.set_epoch(0)
    out = model(patches, pathway_scores, mode="pretrain")
    print(f"[epoch 0 - random neg]")
    print(f"  loss:     {out['loss'].item():.4f}")
    print(f"  loss_itc: {out['loss_itc'].item():.4f}")
    print(f"  loss_itm: {out['loss_itm'].item():.4f}")
    print(f"  loss_mom: {out['loss_mom'].item():.4f}")

    model.set_epoch(40)
    out = model(patches, pathway_scores, mode="pretrain")
    print(f"\n[epoch 40 - hard neg]")
    print(f"  loss:     {out['loss'].item():.4f}")

    out_eval = model(patches, pathway_scores, mode="eval")
    print(f"\n[eval mode]")
    print(f"  cls_p:      {out_eval['cls_p'].shape}")
    print(f"  cls_o:      {out_eval['cls_o'].shape}")
    print(f"  z_wsi:      {out_eval['z_wsi'].shape}")
    print(f"  z_rna:      {out_eval['z_rna'].shape}")
    print(f"  fusion_wsi: {out_eval['fusion_wsi'].shape}")
    print(f"  fusion_rna: {out_eval['fusion_rna'].shape}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n학습 파라미터: {n_params:,}개")