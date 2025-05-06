import os
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from wavie_master.dataset import DeepfakeDataset
from wavie_master.model import WavieModel

from wavie_master.utils import *
from wavie_master.losses import SupConLoss

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

REAL_DIR        = "./data/real"
FAKE_DIR        = "./data/fake"
CHECKPOINT_DIR     = "./checkpoints"
EPOCHS             = 10
BATCH_SIZE         = 32
LEARNING_RATE      = 1e-3
CONTRASTIVE_WEIGHT = 0.1
VAL_INTERVAL       = 1
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
PROJ_DIM    = 1024
N_PROJ         = 1

def main():
    # 0) Model, losses, optimizer, scaler
    model = WavieModel(proj_dim=PROJ_DIM,nproj=N_PROJ, device=DEVICE)

    #BCE for final logit and SupCon for learned embeddings

    bce_loss = torch.nn.BCEWithLogitsLoss()
    supcon_loss = SupConLoss()

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    scaler = GradScaler()

    # 1) Data Loading: We do the preprocessing of CLIP to normalize/resize the images after transforms from the dataset.py file
    transform = model.clip_preprocess

    train_ds = DeepfakeDataset(realLoc=REAL_DIR,fakeLoc=FAKE_DIR,split = "train", transform=transform)
    val_ds = DeepfakeDataset(realLoc=REAL_DIR,fakeLoc=FAKE_DIR,split = "val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    best_val_acc = 0.0
    best_epoch = 0

    # 2) Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = running_cls = running_con = 0.0
        correct = total = 0

        train_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")

        for images, labels in train_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

    # 3) Forward pass
    with autocast():
        logits, embeddings = model(images)

        # 4) Classification loss (BCE on the single logit)
        cls = bce_loss(logits, labels)

        # 5) Contrastive loss on the final embedding
        if CONTRASTIVE_WEIGHT > 0:
            feats_norm = F.normalize(embeddings, dim=1)
            con = supcon_loss(feats_norm.unsqueeze(1), labels.long().view(-1))
        else:
            con = torch.tensor(0.0, device=DEVICE)

        loss = cls + CONTRASTIVE_WEIGHT * con

        # 6) Backward + optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 7) Tracking metrics
        running_loss += loss.item() * images.size(0)
        running_cls  += cls.item()  * images.size(0)
        running_con  += con.item()  * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += images.size(0)

        train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls":  f"{cls.item():.4f}",
                "con":  f"{con.item():.4f}",
                "acc":  f"{correct/total:.4f}"
            })
    train_bar.close()

    # Epoch-level averages
    avg_loss = running_loss / total
    avg_cls  = running_cls  / total
    avg_con  = running_con  / total
    train_acc  = correct / total

    # 8) Validation
    val_acc = val_loss = None

    if epoch % VAL_INTERVAL == 0:
        model.eval()

        if epoch % VAL_INTERVAL == 0:
            model.eval()
            v_corr = v_tot = v_loss_sum = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch}/{EPOCHS}")
                for images, labels in val_bar:
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True).float().view(-1, 1)
                    with autocast():
                        logits, _ = model(images)
                        cls_val = bce_loss(logits, labels)
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    v_corr   += (preds == labels.long()).sum().item()
                    v_tot    += images.size(0)
                    v_loss_sum += cls_val.item() * images.size(0)
                val_bar.close()

            val_acc  = v_corr / v_tot
            val_loss = v_loss_sum / v_tot

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                save_checkpoint(
                    model, optimizer, epoch,
                    os.path.join(CHECKPOINT_DIR, f"best_model_{PROJ_DIM}_{N_PROJ}.pth")
                )

            # 9) Epoch summary and others
            print(
            f"Epoch {epoch}: "
            f"Train Loss={avg_loss:.4f}(Cls={avg_cls:.4f},Con={avg_con:.4f}) "
            f"Train Acc={train_acc:.4f}",
            end=""
            )
            if val_acc is not None:
                print(
                    f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                    + (" [BEST]" if epoch == best_epoch else "")
                )
            else:
                print()

            # 10) Save last checkpoint per epoch
            save_checkpoint(
            model, optimizer, epoch,
            os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_{N_PROJ}_{PROJ_DIM}.pth")
            )

    print(
        f"\nTraining finished! Best Val Acc={best_val_acc:.4f} at epoch {best_epoch}."
    )

if __name__ == "__main__":
    main()