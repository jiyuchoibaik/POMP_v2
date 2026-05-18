# POMP_v2/pretraining/multimodal_encoder.py

"""
멀티모달 인코더 (수정본)
────────────────────────────────────────────────────────
변경사항:
  - ITC projection을 pomp.py로 이동 (mm_encoder에서 제거)
  - BidirectionalCrossAttention에 wsi_proj, rna_proj 추가
  - CrossAttention → FFN → residual 구조 유지

흐름:
  WSI tokens (B, N+1, 512) + RNA tokens (B, 51, 512)
    → 양방향 Cross-Attention
    → FeedForward (pure MLP, no self-attention)
    → ITM head: concat(H_wsi_cls, H_rna_cls) → binary classification
    → MOM head: masked position에만 MSE 복원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# FeedForward Block
# ══════════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, dim: int = 512, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


# ══════════════════════════════════════════════════════════════════════════════
# 양방향 Cross-Attention 블록
# ══════════════════════════════════════════════════════════════════════════════
class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # pre-norm
        self.pre_norm_wsi = nn.LayerNorm(dim, eps=1e-6)
        self.pre_norm_rna = nn.LayerNorm(dim, eps=1e-6)

        # WSI → RNA 방향 (Q=WSI, K/V=RNA)
        self.wsi_q  = nn.Linear(dim, dim)
        self.rna_kv = nn.Linear(dim, dim * 2)
        self.wsi_proj = nn.Linear(dim, dim)   # ← output projection

        # RNA → WSI 방향 (Q=RNA, K/V=WSI)
        self.rna_q  = nn.Linear(dim, dim)
        self.wsi_kv = nn.Linear(dim, dim * 2)
        self.rna_proj = nn.Linear(dim, dim)   # ← output projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # post cross-attention norm (residual 포함)
        self.norm_wsi = nn.LayerNorm(dim, eps=1e-6)
        self.norm_rna = nn.LayerNorm(dim, eps=1e-6)

    def _reshape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return x.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, wsi_tokens: torch.Tensor, rna_tokens: torch.Tensor) -> tuple:
        B, N, _ = wsi_tokens.shape
        _, M, _ = rna_tokens.shape

        # pre-norm
        wsi_normed = self.pre_norm_wsi(wsi_tokens)
        rna_normed = self.pre_norm_rna(rna_tokens)

        # ── WSI → RNA: Q=WSI, K/V=RNA ────────────────────────────────────
        wsi_q        = self._reshape(self.wsi_q(wsi_normed), B, N)
        rna_k, rna_v = self.rna_kv(rna_normed).chunk(2, dim=-1)
        rna_k        = self._reshape(rna_k, B, M)
        rna_v        = self._reshape(rna_v, B, M)

        attn_wsi = (wsi_q @ rna_k.transpose(-2, -1)) * self.scale
        attn_wsi = self.attn_drop(attn_wsi.softmax(dim=-1))
        H_wsi    = (attn_wsi @ rna_v).transpose(1, 2).reshape(B, N, -1)
        H_wsi    = self.norm_wsi(self.proj_drop(self.wsi_proj(H_wsi)) + wsi_tokens)

        # ── RNA → WSI: Q=RNA, K/V=WSI ────────────────────────────────────
        rna_q        = self._reshape(self.rna_q(rna_normed), B, M)
        wsi_k, wsi_v = self.wsi_kv(wsi_normed).chunk(2, dim=-1)
        wsi_k        = self._reshape(wsi_k, B, N)
        wsi_v        = self._reshape(wsi_v, B, N)

        attn_rna = (rna_q @ wsi_k.transpose(-2, -1)) * self.scale
        attn_rna = self.attn_drop(attn_rna.softmax(dim=-1))
        H_rna    = (attn_rna @ wsi_v).transpose(1, 2).reshape(B, M, -1)
        H_rna    = self.norm_rna(self.proj_drop(self.rna_proj(H_rna)) + rna_tokens)

        return H_wsi, H_rna


# ══════════════════════════════════════════════════════════════════════════════
# 멀티모달 인코더
# ══════════════════════════════════════════════════════════════════════════════
class MultimodalEncoder(nn.Module):
    """
    입력:
        wsi_tokens:    (B, N+1, 512)
        rna_tokens:    (B, 51,  512)
        rna_mask:      (B, 50) bool   MOM 마스킹 (True=마스킹)
        target_scores: (B, 50)        MOM 복원 타겟

    출력: dict
        H_wsi_cls:  (B, 512)
        H_rna_cls:  (B, 512)
        itm_logits: (B, 2)
        loss_mom:   scalar or None
        fusion_wsi: (B, N+1, 512)
        fusion_rna: (B, 51,  512)
    """
    def __init__(
        self,
        dim: int           = 512,
        cls_dim: int       = 256,   # 사용하지 않지만 호환성 유지
        num_heads: int     = 8,
        depth: int         = 2,
        mlp_ratio: float   = 4.0,
        drop_rate: float   = 0.1,
        attn_drop_rate: float = 0.1,
        num_pathways: int  = 50,
    ):
        super().__init__()

        self.num_pathways = num_pathways

        # ── 양방향 Cross-Attention ────────────────────────────────────────
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
            )
            for _ in range(depth)
        ])

        # ── FeedForward ───────────────────────────────────────────────────
        self.wsi_ffn = nn.ModuleList([
            FeedForward(dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.rna_ffn = nn.ModuleList([
            FeedForward(dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])

        # ── ITM head ──────────────────────────────────────────────────────
        self.itm_head = nn.Sequential(
            nn.LayerNorm(dim * 2, eps=1e-6),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim, 2),
        )

        # ── MOM head ──────────────────────────────────────────────────────
        self.mom_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        # MOM 마스킹 토큰
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        wsi_tokens:    torch.Tensor,
        rna_tokens:    torch.Tensor,
        rna_mask:      torch.Tensor = None,
        target_scores: torch.Tensor = None,
    ) -> dict:

        B = wsi_tokens.shape[0]

        # ── MOM: 마스킹 적용 ──────────────────────────────────────────────
        if rna_mask is not None:
            rna_tokens = rna_tokens.clone()
            mask_expanded = rna_mask.unsqueeze(-1).expand_as(rna_tokens[:, 1:, :])
            mask_token    = self.mask_token.expand(B, self.num_pathways, -1)
            rna_tokens[:, 1:, :] = torch.where(
                mask_expanded, mask_token, rna_tokens[:, 1:, :]
            )

        # ── CrossAttention → FeedForward ──────────────────────────────────
        H_wsi = wsi_tokens
        H_rna = rna_tokens

        for cross_attn, wsi_ffn, rna_ffn in zip(
            self.cross_attn_layers, self.wsi_ffn, self.rna_ffn
        ):
            H_wsi, H_rna = cross_attn(H_wsi, H_rna)
            H_wsi = wsi_ffn(H_wsi)
            H_rna = rna_ffn(H_rna)

        # ── ITM ───────────────────────────────────────────────────────────
        H_wsi_cls = H_wsi[:, 0, :]   # (B, 512)
        H_rna_cls = H_rna[:, 0, :]   # (B, 512)

        itm_logits = self.itm_head(
            torch.cat([H_wsi_cls, H_rna_cls], dim=-1)
        )                             # (B, 2)

        # ── MOM ───────────────────────────────────────────────────────────
        loss_mom = None
        if rna_mask is not None and target_scores is not None:
            mom_logits = self.mom_head(H_rna[:, 1:, :]).squeeze(-1)  # (B, 50)
            loss_mom   = F.mse_loss(
                mom_logits[rna_mask],
                target_scores[rna_mask],
            )

        return {
            "H_wsi_cls":  H_wsi_cls,
            "H_rna_cls":  H_rna_cls,
            "itm_logits": itm_logits,
            "loss_mom":   loss_mom,
            "fusion_wsi": H_wsi,      # (B, N+1, 512)
            "fusion_rna": H_rna,      # (B, 51,  512)
        }


# ══════════════════════════════════════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    B, N = 4, 100

    wsi_tokens    = torch.rand(B, N + 1, 512)
    rna_tokens    = torch.rand(B, 51, 512)
    rna_mask      = torch.rand(B, 50) < 0.15
    target_scores = torch.rand(B, 50)

    encoder = MultimodalEncoder()
    out = encoder(wsi_tokens, rna_tokens, rna_mask, target_scores)

    print(f"H_wsi_cls:  {out['H_wsi_cls'].shape}")    # (4, 512)
    print(f"H_rna_cls:  {out['H_rna_cls'].shape}")    # (4, 512)
    print(f"itm_logits: {out['itm_logits'].shape}")   # (4, 2)
    print(f"loss_mom:   {out['loss_mom'].item():.4f}")
    print(f"fusion_wsi: {out['fusion_wsi'].shape}")   # (4, 101, 512)
    print(f"fusion_rna: {out['fusion_rna'].shape}")   # (4, 51,  512)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n학습 파라미터: {n_params:,}개")