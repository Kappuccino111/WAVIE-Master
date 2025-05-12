import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import clip  # OpenAI CLIP
except ImportError as e:
    # If the CLIP package is not installed, raise error (see README for installation)
    raise e

from pytorch_wavelets import DWT1DForward, DWT1DInverse, DWTInverse


# What we aim to do:
'''
Input Image
    ↓
CLIP Backbone (frozen, hooks capture CLS outputs)
    ↓
Stacked CLS Tokens from all transformer layers
    ↓
proj1 → per-layer projection
    ↓
Trainable Fusion (alpha weights + softmax + sum)
    ↓
DWT (wavelet decomposition)
    ↓
Refine Low-Frequency Component
    ↓
Inverse DWT (reconstruct)
    ↓
proj2 → fuse features deeper
    ↓
Final Classifier → Single Logit (Fake or Real)
'''

# From RINE
class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

# Mixture of RINE and Wavelet-CLIP
class WavieModel(nn.Module):
    """
    Model integrating CLIP backbone, trainable layer-fusion (RINE), and wavelet refinement (Wavelet-CLIP).
    Outputs a single logit indicating fake (1) or real (0).
    """
    def __init__(self, proj_dim=1024, nproj=1, device='cuda'):
        super(WavieModel, self).__init__()

        self.to(device)
        self.device = device

        self.proj_dim = proj_dim
        self.nproj = nproj

        # Load CLIP model/backbone and freeze parameters
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip_model.visual.named_modules()
            if "ln_2" in name
        ]

        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))

        proj1_layers = [nn.Dropout(p=0.1)]

        base_dim = self.clip_model.visual.proj.shape[0]

        for i in range(nproj):
            in_dim = proj_dim if i > 0 else base_dim
            proj1_layers += [
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)
            ]
        self.proj1 = nn.Sequential(*proj1_layers)

        # Second projection MLP (proj2) for fused feature
        proj2_layers = [nn.Dropout(p=0.1)]
        for i in range(nproj):
            proj2_layers += [
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)
            ]
        self.proj2 = nn.Sequential(*proj2_layers)

        # Wavelet transform modules (6-wavelet , 3-level)
        self.dwt = DWT1DForward(wave='db6', J=3)
        self.idwt = DWT1DInverse(wave='db6')

        # Linear layer for low-frequency part (137 is length of low-freq for 1024 input with db6, J=3)
        low_dim = 137
        self.slp = nn.Linear(low_dim, low_dim)

        # Final classifier head (linear to 1 logit)
        self.classifier = nn.Linear(proj_dim, 1)

        # Move entire model with new parameters to device
        self.to(device)
        self.device = device

    def forward(self, x):
        """
        Forward pass: returns (logit, feature_vector).
        - x: input images tensor of shape [N, 3, H, W].
        """

        x = x.to(self.device)
        N = x.size(0)

        # 1) CLIP forward + hooks
        with torch.no_grad():
            _ = self.clip_model.encode_image(x)

        # 2) collect CLS tokens → [N, L, embed_dim]
        cls_tokens = []

        for h in self.hooks:
            out = h.output
            # only keep 3-D activations
            if out is None or out.dim() != 3:
                continue

            if out.shape[0] == N:
                # case A: straightforward
                token = out[:, 0, :]      # [batch, dim]
            elif out.shape[1] == N:
                # case B: swap axes then grab
                token = out[0, :, :]      # [batch, dim]
            else:
                # something unexpected—skip
                continue

            cls_tokens.append(token)

        if len(cls_tokens) == 0:
            raise RuntimeError("No features captured from CLIP hooks.")

        # Stack features from all layers: shape [N, L, embed_dim]
        g = torch.stack(cls_tokens, dim=1)

        # 3) per-layer projection        ← proj1
        g_proj = self.proj1(g)  # [N, L, proj_dim]

        # 4) trainable fusion
        alpha_weights = torch.softmax(self.alpha, dim=1)  # [1, L, proj_dim]
        fused = alpha_weights * g_proj                   # broadcast to [N, L, proj_dim]
        z = fused.sum(dim=1)                             # fused feature [N, proj_dim]

        # 5) wavelet decomposition
        yl, yh = self.dwt(z.unsqueeze(1)) # yl:[N,1,low_dim], yh:list
        yl = yl.view(yl.size(0), -1)      # [N,low_dim]

        # 6) refine low-frequency
        yl_new = self.slp(yl)             # [N,low_dim]
        yl_new = yl_new.view(yl_new.size(0), 1, -1)

        # 7) inverse DWT (reconstruct)
        x_refined = self.idwt((yl_new, yh))
        x_refined = x_refined.squeeze(1)  # [N, proj_dim]

        # 8) second projection MLP (proj2)
        x_refined = self.proj2(x_refined) # [N, proj_dim]

        # 9) final classifier
        logit = self.classifier(x_refined)  # [N,1]

        return logit, x_refined