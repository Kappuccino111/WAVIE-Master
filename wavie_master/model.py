import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import clip  # OpenAI CLIP
except ImportError as e:
    # If the CLIP package is not installed, raise error (see README for installation)
    raise e

from pytorch_wavelets import DWTForward, DWT1DInverse, DWTInverse

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
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))

        proj1_layers = [nn.Dropout(p=0.1)]
        for i in range(nproj):
            in_dim = proj_dim if i > 0 else self.clip_model.visual.width
            proj1_layers += [
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)
            ]
        self.proj1 = nn.Sequential(*proj1_layers)
