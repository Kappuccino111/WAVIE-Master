import os
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score

from wavie_master.dataset import DeepfakeDataset
from wavie_master.model import WavieModel
from train import PROJ_DIM, N_PROJ
from wavie_master.utils import load_checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

REAL_DIR        = "./data/real"
FAKE_DIR        = "./data/fake"
CHECKPOINT_PATH = "./checkpoints/best_model_{N_PROJ}_{PROJ_DIM}.pth"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 128

def main(loc):
    # 0) Build model and load weights
    model = WavieModel(proj_dim=PROJ_DIM, nproj=N_PROJ, device=DEVICE)
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    # 1) Data loading (use same CLIP preprocessing)
    transform  = model.clip_preprocess
    test_ds    = DeepfakeDataset(realLoc=REAL_DIR,
                                 fakeLoc=FAKE_DIR,
                                 split="test",
                                 transform=transform)
    test_loader = DataLoader(test_ds,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=12,
                             pin_memory=True)

    # 2) Loss and accumulators
    bce_loss      = nn.BCEWithLogitsLoss()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # to compute AUC
    all_probs = []
    all_labels = []

    # 3) Inference loop
    with torch.no_grad():
        bar = tqdm(test_loader, desc="[Test]")
        for images, labels in bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).float().view(-1,1)

            with autocast():
                logits, _ = model(images)
                loss = bce_loss(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            batch_size = images.size(0)
            total_loss    += loss.item() * batch_size
            total_correct += (preds == labels.long()).sum().item()
            total_samples += batch_size

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        bar.close()

    # 4) Compute metrics
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"\nTest Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}")

    # 5) AUC
    try:
        y_true  = torch.cat(all_labels).numpy()
        y_score = torch.cat(all_probs).numpy()
        auc = roc_auc_score(y_true, y_score)
        print(f"Test AUC: {auc:.4f}")
    except ImportError:
        pass

if __name__ == "__main__":
    main()