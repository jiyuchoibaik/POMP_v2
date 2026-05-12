# POMP_v2/pretraining/rna_encoder.py

"""
RNA 인코더
────────────────────────────────────────────────────────
흐름:
  pathway_scores (B, 50)
    → score embedding: Linear(1, 512)       ← 활성도 정보
    + pathway identity embedding: (50, 512) ← pathway 종류 정보
    → [CLS_O, p1, ..., p50]
    → Omics-Transformer (depth=2)
    → CLS_O (B, 256)
    → tokens (B, 51, 512)
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block


class RNAEncoder(nn.Module):
    """
    RNA 인코더.

    입력: (B, 50)  ssGSEA Hallmark pathway 점수
    출력:
        cls:    (B, 256)       CLS_O → ITC loss, 기하학적 분석
        tokens: (B, 51, 512)  전체 토큰 → Cross-Attention
    """
    def __init__(
        self,
        num_pathways: int = 50,
        hidden_dim: int = 512,
        output_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
    ):
        super().__init__()

        self.num_pathways = num_pathways

        # ── score embedding: 활성도 정보 ─────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
        )

        # ── pathway identity embedding: pathway 종류/semantic 정보 ───────
        # 각 pathway가 고유한 semantic prior를 갖도록
        # (EMT, MYC, Hypoxia, ... 각각 다른 embedding)
        self.pathway_embed = nn.Parameter(
            torch.randn(num_pathways, hidden_dim)
        )
        nn.init.trunc_normal_(self.pathway_embed, std=0.02)

        # ── CLS 토큰 ──────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # ── CLS → output_dim ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (B, 50)
        returns:
            cls:    (B, 256)
            tokens: (B, 51, 512)
        """
        B = x.shape[0]

        # score embedding
        x = x.unsqueeze(-1)                              # (B, 50, 1)
        x = self.input_proj(x)                           # (B, 50, 512)

        # pathway identity embedding 결합
        # 각 token = "이 pathway의 활성도" + "이 pathway가 무엇인지"
        x = x + self.pathway_embed.unsqueeze(0)          # (B, 50, 512)

        # CLS prepend
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, 512)
        x = torch.cat([cls, x], dim=1)                   # (B, 51, 512)

        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                 # (B, 51, 512)

        # CLS projection
        cls_out = self.head(x[:, 0, :])                  # (B, 256)

        return cls_out, x


# ══════════════════════════════════════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    B = 4
    pathway_scores = torch.rand(B, 50)

    encoder = RNAEncoder()
    cls, tokens = encoder(pathway_scores)

    print(f"입력: {pathway_scores.shape}")
    print(f"CLS_O:  {cls.shape}")       # (4, 256)
    print(f"tokens: {tokens.shape}")    # (4, 51, 512)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"학습 파라미터: {n_params:,}개")