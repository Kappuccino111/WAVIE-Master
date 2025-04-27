import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss. Supports supervised and unsupervised contrastive learning."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Compute contrastive loss.
        - features: tensor of shape [batch_size, n_views, feature_dim]
        - labels: tensor of shape [batch_size] (optional)
        - mask: contrastive mask of shape [batch_size, batch_size] (optional)
        If both labels and mask are None, defaults to unsupervised (SimCLR) loss.
        """
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [batch_size, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        # Default mask for unsupervised: identity matrix
        if labels is None and mask is None:
            mask = torch.eye(batch_size, device=device)
        if labels is not None:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        if mask is not None:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]

        # Flatten out all views to compute pairwise similarities
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute logits (anchor dot contrast features divided by temperature)
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # For numerical stability, subtract max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Contrastive mask: mask out self-comparisons
        mask = mask.repeat(anchor_count, contrast_count)
        self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * self_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-likelihood for positive pairs
        positive_mask = mask
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-12)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
