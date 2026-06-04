class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.3):
        """
        CRITICAL FIX: alpha=0.75 (focus on fire class, not background)
        dice_weight=0.3 (reduced to prevent dice from dominating when no predictions)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Focal loss with CORRECT alpha (0.75 = focus on fire pixels)
        bce_loss = self.bce(logits, targets)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = (alpha_t * (1.0 - p_t) ** self.gamma * bce_loss).mean()

        # Dice loss with higher smooth to prevent division issues
        smooth = 1.0  # Increased from 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()

        return (1.0 - self.dice_weight) * focal + self.dice_weight * dice


class PositivePenaltyLoss(nn.Module):
    """
    NEW: Penalizes the model for predicting all zeros.
    This forces the model to predict SOMETHING for fire pixels.
    """
    def __init__(self, min_fire_prob=0.1):
        super().__init__()
        self.min_fire_prob = min_fire_prob

    def forward(self, probs, targets):
        # Only apply on samples that have fire
        fire_mask = (targets.sum(dim=(1,2,3)) > 0).float()
        if fire_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)

        # For pixels that SHOULD be fire, penalize low probabilities
        fire_pixels = targets * probs
        fire_penalty = F.relu(self.min_fire_prob - fire_pixels).mean()

        return fire_penalty * fire_mask.mean()


class CombinedLoss(nn.Module):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=0.3,
                 edge_weight=0.05, positive_weight=0.2):
        super().__init__()
        self.focal_dice = FocalDiceLoss(focal_alpha, focal_gamma, dice_weight)
        self.edge_loss = SmoothEdgeLoss()
        self.positive_loss = PositivePenaltyLoss(min_fire_prob=0.3)
        self.edge_weight = edge_weight
        self.positive_weight = positive_weight

    def forward(self, logits, targets):
        loss_fd = self.focal_dice(logits, targets)
        probs = torch.sigmoid(logits)
        loss_edge = self.edge_loss(probs, targets)
        loss_pos = self.positive_loss(probs, targets)

        return loss_fd + self.edge_weight * loss_edge + self.positive_weight * loss_pos
