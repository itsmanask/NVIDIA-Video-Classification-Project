"""
model.py

Defines VideoClassifier — a ResNet18-based video classification model.

Architecture:
    Input  : (B, 16, 3, 224, 224)
    Step 1 : Reshape → (B×16, 3, 224, 224)
    Step 2 : ResNet18 backbone (pretrained) → (B×16, 512)
    Step 3 : Reshape → (B, 16, 512)
    Step 4 : Temporal average pool → (B, 512)      ← mean across 16 frames
    Step 5 : Dropout(0.5)
    Step 6 : Linear(512 → 256) + ReLU
    Step 7 : Dropout(0.3)
    Step 8 : Linear(256 → 4)                       ← raw logits, not softmax
    Output : (B, 4)

Why ResNet18 + average pooling:
    - Pretrained on ImageNet — already encodes edges, textures, shapes,
      which are the primary discriminators for Animation / Gaming /
      Natural_Content / Flat_Content
    - Average pooling across 16 frames is sufficient for these 4 classes
      because each class is visually distinct at the frame level, not just
      at the temporal-sequence level. An LSTM would add complexity without
      meaningful accuracy gain here.
    - ResNet18 fits comfortably inside a 10 GB MIG partition (A100 1g.10gb)
      with batch_size=32 in frozen-backbone phase.

Two-phase training strategy:
    Phase 1 — Freeze backbone, train classifier head only
        - Fast convergence (fewer trainable params)
        - Safer: pretrained features stay intact during early epochs
        - Recommended: 10–15 epochs
        - Call model.freeze_backbone() before phase 1 training

    Phase 2 — Unfreeze last 2 ResNet blocks, fine-tune end-to-end
        - Allows backbone to adapt to your specific visual domain
        - Use a lower learning rate (≈ 10× smaller than phase 1)
        - Recommended: 5–10 additional epochs
        - Call model.unfreeze_last_blocks(n=2) before phase 2 training

    This is managed in train.py — model.py only provides the methods.

Label mapping (fixed forever):
    0 → Animation
    1 → Flat_Content
    2 → Gaming
    3 → Natural_Content

Usage:
    from src.model import VideoClassifier, build_model
    model = build_model(num_classes=4, freeze_backbone=True)
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Number of Classes ─────────────────────────────────────────────────
NUM_CLASSES = 4


# ── Model ─────────────────────────────────────────────────────────────
class VideoClassifier(nn.Module):
    """
    ResNet18-based video classifier with temporal average pooling.

    Args:
        num_classes    : number of output classes (default: 4)
        freeze_backbone: if True, ResNet18 weights are frozen on init
                         Call unfreeze_last_blocks() or unfreeze_backbone()
                         later to enable fine-tuning
        dropout_1      : dropout rate before the first linear layer (default: 0.5)
        dropout_2      : dropout rate before the final linear layer (default: 0.3)
    """

    def __init__(
        self,
        num_classes     = NUM_CLASSES,
        freeze_backbone = True,
        dropout_1       = 0.5,
        dropout_2       = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ── Backbone: ResNet18 pretrained on ImageNet ─────────────────
        # We remove the final FC layer (resnet.fc) so the backbone
        # outputs a (B*16, 512) feature vector per frame.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Drop the final classification layer
        # resnet.fc was Linear(512, 1000) — we replace it with Identity
        # so the backbone outputs raw 512-dim feature vectors
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output shape after backbone: (B*16, 512, 1, 1)
        # After squeeze: (B*16, 512)

        # ── Classifier Head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_2),
            nn.Linear(256, num_classes),
            # No softmax here — CrossEntropyLoss expects raw logits
        )

        # ── TASK 1: Temporal Attention ────────────────────────────────
        # Replaces mean pooling. Learns a scalar weight per frame so
        # the model can upweight informative frames and ignore redundant
        # ones (e.g. near-identical frames in Flat_Content videos).
        # bias=False: only one param per 512-dim input — ultra-lightweight
        self.attention = nn.Linear(512, 1, bias=False)

        # ── Phase 1: Freeze backbone by default ───────────────────────
        if freeze_backbone:
            self.freeze_backbone()

    # ── Forward Pass ──────────────────────────────────────────────────
    def forward(self, x):
        """
        Args:
            x : FloatTensor of shape (B, T, C, H, W)
                B = batch size
                T = number of frames (16)
                C = channels (3)
                H = height (224)
                W = width (224)

        Returns:
            logits : FloatTensor of shape (B, num_classes)
                     Raw scores — NOT probabilities
                     Pass to CrossEntropyLoss during training
                     Pass to softmax for inference probabilities
        """
        B, T, C, H, W = x.shape

        # ── Step 1: Flatten batch and time dimensions ─────────────────
        # Treat every frame of every video as an independent image
        # (B, 16, 3, 224, 224) → (B*16, 3, 224, 224)
        x = x.reshape(B * T, C, H, W)  # FIX 1: reshape() handles non-contiguous tensors safely

        # ── Step 2: Run all frames through ResNet18 backbone ──────────
        # (B*16, 3, 224, 224) → (B*16, 512, 1, 1)
        x = self.backbone(x)

        # ── Step 3: Remove spatial singleton dimensions ───────────────
        # (B*16, 512, 1, 1) → (B*16, 512)
        x = x.squeeze(-1).squeeze(-1)

        # ── Step 4: Restore batch and time dimensions ─────────────────
        # (B*16, 512) → (B, 16, 512)
        x = x.reshape(B, T, -1)  # FIX 1: reshape() handles non-contiguous tensors safely

        # ── Step 5: Temporal attention pooling ───────────────────────
        # TASK 1: Replaces mean pooling with learned per-frame weights.
        # self.attention projects each 512-dim frame vector → scalar
        # softmax normalises weights across T=16 so they sum to 1
        # weighted sum collapses (B, 16, 512) → (B, 512)
        attn_weights = self.attention(x)                    # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)   # (B, T, 1)
        x = (x * attn_weights).sum(dim=1)                  # (B, 512)

        # ── Step 6: Classify ──────────────────────────────────────────
        # (B, 512) → (B, 4)
        logits = self.classifier(x)

        return logits

    # ── Backbone Control Methods ──────────────────────────────────────
    def freeze_backbone(self):
        """
        Freeze all ResNet18 backbone parameters.
        Only the classifier head will train.

        Use for Phase 1 training.
        Memory efficient — gradients not computed for backbone.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        # FIX 2: Put all BatchNorm2d layers in eval mode so running
        # mean/var stats are frozen and not updated during Phase 1.
        # This only affects backbone — classifier head is untouched.
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        print("  [model] Backbone FROZEN — only classifier head trains")
        self._print_trainable_params()

    def unfreeze_backbone(self):
        """
        Unfreeze all ResNet18 backbone parameters.
        Entire network trains end-to-end.

        Use only if you want full fine-tuning.
        Requires a lower learning rate (≈10× smaller than phase 1).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        # FIX 3: Restore all backbone BatchNorm2d layers to train mode
        # so running stats update again during full fine-tuning.
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()

        print("  [model] Backbone FULLY UNFROZEN — entire network trains")
        self._print_trainable_params()

    def unfreeze_last_blocks(self, n=2):
        """
        Unfreeze the last n ResNet18 layer blocks for fine-tuning.
        Everything else stays frozen.

        Default n=2 unfreezes layer3 and layer4 (the deeper, more
        task-specific layers) while keeping early layers frozen.
        This is the recommended Phase 2 setting.

        ResNet18 block structure:
            [0]  conv1
            [1]  bn1
            [2]  relu
            [3]  maxpool
            [4]  layer1   ← low-level features (edges)
            [5]  layer2   ← mid-level features (textures)
            [6]  layer3   ← high-level features  ← unfrozen with n=2
            [7]  layer4   ← task-specific        ← unfrozen with n=2
            [8]  avgpool

        Args:
            n : number of ResNet blocks to unfreeze from the end
                n=1 → only layer4
                n=2 → layer3 + layer4 (recommended)
                n=4 → layer1..4 (aggressive fine-tuning)
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then unfreeze the last n children blocks
        backbone_children = list(self.backbone.children())
        blocks_to_unfreeze = backbone_children[-n:]

        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        # FIX 3: Restore BatchNorm2d to train mode ONLY in unfrozen blocks.
        # Frozen blocks keep their BatchNorm2d in eval() (set by freeze_backbone).
        for block in blocks_to_unfreeze:
            for module in block.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()

        print(f"  [model] Last {n} backbone blocks UNFROZEN — fine-tuning mode")
        self._print_trainable_params()

    # ── Utility ───────────────────────────────────────────────────────
    def _print_trainable_params(self):
        """Prints trainable vs total parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        frozen    = total - trainable
        print(f"  [model] Parameters — "
              f"total: {total:,}  |  "
              f"trainable: {trainable:,}  |  "
              f"frozen: {frozen:,}")

    def get_info(self):
        """Returns a dict with model configuration details."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {
            'backbone'          : 'ResNet18 (pretrained ImageNet)',
            'temporal_pooling'  : 'attention pooling across 16 frames',  # FIX 6: reflects actual pooling method
            'classifier_head'   : 'Linear(512→256→4)',
            'num_classes'       : self.num_classes,
            'total_params'      : total,
            'trainable_params'  : trainable,
            'frozen_params'     : total - trainable,
        }


# ── Build Function ────────────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES, freeze_backbone=True, device=None):
    """
    Convenience function — builds model, moves to device, prints summary.

    Args:
        num_classes     : number of output classes (default: 4)
        freeze_backbone : freeze ResNet18 backbone on init (default: True)
        device          : torch.device — auto-detects CUDA if None

    Returns:
        model  : VideoClassifier on the specified device
        device : torch.device (useful to store in train.py)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VideoClassifier(
        num_classes     = num_classes,
        freeze_backbone = freeze_backbone,
    ).to(device)

    print(f"\n{'='*60}")
    print("MODEL BUILT")
    print(f"{'='*60}")
    info = model.get_info()
    for k, v in info.items():
        print(f"  {k:<22}: {v}")
    print(f"  {'device':<22}: {device}")
    print(f"{'='*60}\n")

    return model, device


# ── Quick Sanity Test ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("MODEL SANITY CHECK")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Phase 1: Frozen backbone ──────────────────────────────────────
    print("\n--- Phase 1: Frozen backbone ---")
    model, device = build_model(freeze_backbone=True, device=device)

    # Dummy batch: 4 videos × 16 frames × 3 channels × 224×224
    dummy = torch.randn(4, 16, 3, 224, 224).to(device)
    print(f"\nInput shape  : {dummy.shape}")

    with torch.no_grad():
        logits = model(dummy)

    print(f"Output shape : {logits.shape}")         # Expected: (4, 4)
    print(f"Output dtype : {logits.dtype}")          # Expected: float32
    print(f"Sample logits: {logits[0].tolist()}")    # 4 raw scores

    assert logits.shape == (4, NUM_CLASSES), \
        f"Expected (4, {NUM_CLASSES}), got {logits.shape}"

    # ── Phase 2: Unfreeze last 2 blocks ──────────────────────────────
    print("\n--- Phase 2: Unfreeze last 2 blocks ---")
    model.unfreeze_last_blocks(n=2)

    with torch.no_grad():
        logits2 = model(dummy)

    assert logits2.shape == (4, NUM_CLASSES), \
        f"Expected (4, {NUM_CLASSES}), got {logits2.shape}"
    print(f"Output shape after unfreeze: {logits2.shape}")

    # ── Full unfreeze ─────────────────────────────────────────────────
    print("\n--- Full unfreeze ---")
    model.unfreeze_backbone()

    # ── Re-freeze ─────────────────────────────────────────────────────
    print("\n--- Re-freeze ---")
    model.freeze_backbone()

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("  1. Copy this file to:")
    print("     src/model.py")
    print("  2. Run sanity check on the server:")
    print("     python src/model.py")
    print("  3. Confirm output shape is (4, 4) and all checks pass")
    print("  4. Proceed to src/train.py")
    print("=" * 60)
