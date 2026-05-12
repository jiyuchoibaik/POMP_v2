# POMP_v2/pretraining/pomp.py
"""
POMP 전체 모델 (수정본)
────────────────────────────────────────────────────────
수정 사항:
  [1] Loss weight: 1:6:3 → 1:2:1
  [2] Hard negative curriculum (초기: random, 후반: hard)
  [3] eval return 확장 (cls_p, cls_o, fusion_wsi, fusion_rna)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wsi_encoder import WSIEncoder
from .rna_encoder import RNAEncoder
from .multimodal_encoder import MultimodalEncoder


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
        # loss 가중치 (1:2:1로 변경)
        itc_weight: float     = 1.0,
        itm_weight: float     = 2.0,
        mom_weight: float     = 1.0,
        # ITC temperature
        temp: float           = 0.07,
        # hard negative curriculum
        total_epochs: int     = 100,
        hard_negative_start_epoch: int = None,  # None이면 total_epochs*0.3
    ):
        super().__init__()

        self.mask_ratio  = mask_ratio
        self.itc_weight  = itc_weight
        self.itm_weight  = itm_weight
        self.mom_weight  = mom_weight

        # hard negative curriculum
        self.hard_negative_start_epoch = (
            hard_negative_start_epoch
            if hard_negative_start_epoch is not None
            else int(total_epochs * 0.3)
        )
        self.current_epoch = 0   # 학습 루프에서 업데이트

        # learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / temp).log())

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
        """학습 루프에서 매 epoch 시작 시 호출"""
        self.current_epoch = epoch

    # ── ITC loss ──────────────────────────────────────────────────────────
    def _itc_loss(
        self,
        z_wsi: torch.Tensor,
        z_rna: torch.Tensor,
    ) -> torch.Tensor:
        B = z_wsi.shape[0]
        logit_scale = self.logit_scale.exp().clamp(max=100)

        sim_w2r = logit_scale * z_wsi @ z_rna.T   # (B, B)
        sim_r2w = sim_w2r.T

        labels = torch.arange(B, device=z_wsi.device)

        loss_w2r = F.cross_entropy(sim_w2r, labels)
        loss_r2w = F.cross_entropy(sim_r2w, labels)

        return (loss_w2r + loss_r2w) / 2

    # ── Negative index 샘플링 (curriculum) ───────────────────────────────
    def _sample_negatives(
        self,
        z_wsi: torch.Tensor,
        z_rna: torch.Tensor,
    ) -> tuple:
        """
        curriculum:
            current_epoch < hard_negative_start_epoch → uniform random
            current_epoch >= hard_negative_start_epoch → hard negative
        """
        B = z_wsi.shape[0]
        device = z_wsi.device

        if self.current_epoch < self.hard_negative_start_epoch:
            # ── uniform random negative ───────────────────────────────────
            # 자기 자신 제외한 랜덤 인덱스
            neg_rna_idx = torch.zeros(B, dtype=torch.long, device=device)
            neg_wsi_idx = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                candidates = list(range(B))
                candidates.remove(b)
                neg_rna_idx[b] = candidates[torch.randint(len(candidates), (1,)).item()]
                neg_wsi_idx[b] = candidates[torch.randint(len(candidates), (1,)).item()]
        else:
            # ── hard negative mining ──────────────────────────────────────
            with torch.no_grad():
                logit_scale = self.logit_scale.exp().clamp(max=100)
                sim_w2r = logit_scale * z_wsi @ z_rna.T   # (B, B)
                sim_r2w = sim_w2r.T

                weights_w2r = sim_w2r.clone()
                weights_r2w = sim_r2w.clone()
                weights_w2r.fill_diagonal_(float("-inf"))
                weights_r2w.fill_diagonal_(float("-inf"))

                weights_w2r = weights_w2r.softmax(dim=-1)
                weights_r2w = weights_r2w.softmax(dim=-1)

                neg_rna_idx = torch.multinomial(weights_w2r, 1).squeeze(1)
                neg_wsi_idx = torch.multinomial(weights_r2w, 1).squeeze(1)

        return neg_rna_idx, neg_wsi_idx

    # ── ITM + MOM loss ────────────────────────────────────────────────────
    def _itm_mom_loss(
        self,
        wsi_tokens:    torch.Tensor,
        rna_tokens:    torch.Tensor,
        cls_p:         torch.Tensor,
        cls_o:         torch.Tensor,
        z_wsi:         torch.Tensor,
        z_rna:         torch.Tensor,
        target_scores: torch.Tensor,
        rna_mask:      torch.Tensor,
    ) -> tuple:
        B = wsi_tokens.shape[0]
        device = wsi_tokens.device

        # negative 인덱스 샘플링 (curriculum 적용)
        neg_rna_idx, neg_wsi_idx = self._sample_negatives(z_wsi, z_rna)

        # Positive pair (MOM 포함)
        out_pos = self.mm_encoder(
            wsi_tokens, rna_tokens,
            cls_p, cls_o,
            rna_mask=rna_mask,
            target_scores=target_scores,
        )

        # Negative pair A: WSI 고정, RNA hard negative
        out_neg_a = self.mm_encoder(
            wsi_tokens, rna_tokens[neg_rna_idx],
            cls_p, cls_o[neg_rna_idx],
        )

        # Negative pair B: RNA 고정, WSI hard negative
        out_neg_b = self.mm_encoder(
            wsi_tokens[neg_wsi_idx], rna_tokens,
            cls_p[neg_wsi_idx], cls_o,
        )

        # ITM loss
        itm_logits = torch.cat([
            out_pos["itm_logits"],
            out_neg_a["itm_logits"],
            out_neg_b["itm_logits"],
        ], dim=0)                                           # (3B, 2)

        itm_labels = torch.cat([
            torch.ones(B, dtype=torch.long),
            torch.zeros(2 * B, dtype=torch.long),
        ], dim=0).to(device)

        loss_itm = F.cross_entropy(itm_logits, itm_labels)
        loss_mom = out_pos["loss_mom"]

        # fusion representation (positive pair)
        fusion_wsi = out_pos["fusion_wsi"]   # (B, 512)
        fusion_rna = out_pos["fusion_rna"]   # (B, 512)

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

        # ── 단일 모달 인코더 ──────────────────────────────────────────────
        cls_p, wsi_tokens = self.wsi_encoder(patches)
        cls_o, rna_tokens = self.rna_encoder(pathway_scores)

        # ── ITC projection ────────────────────────────────────────────────
        out_mm = self.mm_encoder(wsi_tokens, rna_tokens, cls_p, cls_o)
        z_wsi = out_mm["z_wsi"]       # (B, 256) normalized
        z_rna = out_mm["z_rna"]       # (B, 256) normalized

        if mode == "eval":
            # fusion representation도 함께 반환
            return {
                "cls_p":      cls_p,                  # (B, 256) WSI representation
                "cls_o":      cls_o,                  # (B, 256) RNA representation
                "fusion_wsi": out_mm["fusion_wsi"],   # (B, 512) fusion WSI
                "fusion_rna": out_mm["fusion_rna"],   # (B, 512) fusion RNA
            }

        # ── MOM 마스킹 생성 ───────────────────────────────────────────────
        rna_mask = self._make_rna_mask(B, device)

        # ── ITC loss ──────────────────────────────────────────────────────
        loss_itc = self._itc_loss(z_wsi, z_rna)

        # ── ITM + MOM loss ────────────────────────────────────────────────
        loss_itm, loss_mom, fusion_wsi, fusion_rna = self._itm_mom_loss(
            wsi_tokens, rna_tokens,
            cls_p, cls_o,
            z_wsi, z_rna,
            target_scores=pathway_scores,
            rna_mask=rna_mask,
        )

        # ── 총 loss ───────────────────────────────────────────────────────
        loss = (
            self.itc_weight * loss_itc +
            self.itm_weight * loss_itm +
            self.mom_weight * loss_mom
        )

        return {
            "loss":     loss,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_mom": loss_mom,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    B, N = 2, 50
    patches        = torch.rand(B, N, 3, 256, 256)
    pathway_scores = torch.rand(B, 50)

    model = POMPModel(total_epochs=100)

    # epoch 0: random negative
    model.set_epoch(0)
    out = model(patches, pathway_scores, mode="pretrain")
    print(f"[epoch 0 - random neg]")
    print(f"  loss:     {out['loss'].item():.4f}")
    print(f"  loss_itc: {out['loss_itc'].item():.4f}")
    print(f"  loss_itm: {out['loss_itm'].item():.4f}")
    print(f"  loss_mom: {out['loss_mom'].item():.4f}")

    # epoch 40: hard negative
    model.set_epoch(40)
    out = model(patches, pathway_scores, mode="pretrain")
    print(f"\n[epoch 40 - hard neg]")
    print(f"  loss:     {out['loss'].item():.4f}")

    # eval mode
    out_eval = model(patches, pathway_scores, mode="eval")
    print(f"\n[eval mode]")
    print(f"  cls_p:      {out_eval['cls_p'].shape}")       # (2, 256)
    print(f"  cls_o:      {out_eval['cls_o'].shape}")       # (2, 256)
    print(f"  fusion_wsi: {out_eval['fusion_wsi'].shape}")  # (2, 512)
    print(f"  fusion_rna: {out_eval['fusion_rna'].shape}")  # (2, 512)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n학습 파라미터: {n_params:,}개")