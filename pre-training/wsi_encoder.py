# POMP_v2/pretraining/wsi_encoder.py

"""
WSI 인코더 (UNI2-h 기반)
────────────────────────────────────────────────────────
흐름:
  patches (B, N, 3, 256, 256)
    → UNI2-h ViT (freeze, dim=1536)
    → 패치별 feature (B, N, 1536)
    → Pat-Transformer (CLS + v1,...,vN)
    → CLS_P (B, 256)  ← ITC/기하학적 분석
    → tokens (B, N+1, 512)  ← Cross-Attention

의존성:
  pip install timm huggingface_hub h5py
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from huggingface_hub import hf_hub_download
import timm
import os


# ══════════════════════════════════════════════════════════════════════════════
# UNI2-h ViT (freeze)
# ══════════════════════════════════════════════════════════════════════════════
class UNI2Encoder(nn.Module):
    """
    UNI2-h ViT-H/14 로드 및 feature 추출.
    학습 중 freeze, 패치별 CLS 토큰 추출.

    입력: (B*N, 3, 256, 256)  → UNI2 transform 적용 후 입력
    출력: (B*N, 1536)
    """
    def __init__(self, local_dir: str = "./assets/uni2"):
        super().__init__()

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        # 가중치 로드
        ckpt_path = os.path.join(local_dir, "pytorch_model.bin")
        if not os.path.exists(ckpt_path):
            print("[INFO] UNI2-h 가중치 다운로드 중...")
            os.makedirs(local_dir, exist_ok=True)
            hf_hub_download(
                "MahmoodLab/UNI2-h",
                filename="pytorch_model.bin",
                local_dir=local_dir,
            )

        self.model = timm.create_model(
            "vit_giant_patch14_224",
            pretrained=False,
            **timm_kwargs
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        print("[INFO] UNI2-h 가중치 로드 완료")

        # 완전 freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # UNI2 전용 transform (224×224 리사이즈 + ImageNet 정규화)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, 3, 256, 256) float32, 값 범위 [0, 1]
        returns: (B*N, 1536)
        """
        x = self.transform(x)          # (B*N, 3, 224, 224)
        with torch.no_grad():
            feat = self.model(x)       # (B*N, 1536)
        return feat


# ══════════════════════════════════════════════════════════════════════════════
# Pat-Transformer
# ══════════════════════════════════════════════════════════════════════════════
class PatTransformer(nn.Module):
    """
    슬라이드 수준 Transformer.
    UNI2 feature 집합 → CLS_P (슬라이드 표현)

    입력: (B, N, 1536)
    출력:
        cls:    (B, 256)        CLS_P → ITC loss, 기하학적 분석
        tokens: (B, N+1, 512)  전체 토큰 → Cross-Attention
    """
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        output_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
    ):
        super().__init__()

        # 1536 → 512 projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
        )

        # CLS 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)


        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # CLS → output_dim projection
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (B, N, 1536)
        returns:
            cls:    (B, 256)
            tokens: (B, N+1, 512)
        """
        B, N, _ = x.shape

        # 1536 → 512
        x = self.input_proj(x)                          # (B, N, 512)

        # CLS prepend
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, 512)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, 512)


        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                # (B, N+1, 512)

        # CLS projection
        cls_out = self.head(x[:, 0, :])                 # (B, 256)

        return cls_out, x


# ══════════════════════════════════════════════════════════════════════════════
# WSI 인코더 (전체)
# ══════════════════════════════════════════════════════════════════════════════
class WSIEncoder(nn.Module):
    """
    WSI 인코더 전체 파이프라인.

    입력: patches (B, N, 3, 256, 256)  float32, [0,1]
    출력:
        cls:    (B, 256)        CLS_P
        tokens: (B, N+1, 512)  전체 토큰
    """
    def __init__(
        self,
        uni2_local_dir: str = "./assets/uni2",
        pat_hidden_dim: int = 512,
        pat_output_dim: int = 256,
        pat_depth: int = 2,
        pat_num_heads: int = 8,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.uni2 = UNI2Encoder(local_dir=uni2_local_dir)
        self.pat = PatTransformer(
            input_dim=1536,
            hidden_dim=pat_hidden_dim,
            output_dim=pat_output_dim,
            depth=pat_depth,
            num_heads=pat_num_heads,
            drop_rate=drop_rate,
        )

    def forward(self, patches: torch.Tensor) -> tuple:
        """
        patches: (B, N, 3, 256, 256)
        returns:
            cls:    (B, 256)
            tokens: (B, N+1, 512)
        """
        B, N, C, H, W = patches.shape

        # UNI2: 패치별 독립 처리 (freeze)
        patches_flat = patches.reshape(B * N, C, H, W)     # (B*N, 3, 256, 256)
        features = self.uni2(patches_flat)               # (B*N, 1536)
        features = features.reshape(B, N, -1)               # (B, N, 1536)

        # Pat-Transformer
        cls, tokens = self.pat(features)                 # (B,256), (B,N+1,512)

        return cls, tokens


# ══════════════════════════════════════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    B, N = 2, 50
    patches = torch.rand(B, N, 3, 256, 256)

    # UNI2 가중치 없이 구조만 테스트
    encoder = WSIEncoder.__new__(WSIEncoder)
    encoder.__init__.__func__  # skip

    # Pat-Transformer만 테스트
    pat = PatTransformer()
    dummy_feat = torch.rand(B, N, 1536)
    cls, tokens = pat(dummy_feat)

    print(f"입력 feature: {dummy_feat.shape}")
    print(f"CLS_P:  {cls.shape}")      # (2, 256)
    print(f"tokens: {tokens.shape}")   # (2, 51, 512)