import torch
import torch.nn as nn
import torch.nn.functional as F



# CommonFeatureEnhance Loss
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, featureA, featureB):
        total_loss = 0
        device = (torch.device('cuda') if featureA.is_cuda else torch.device('cpu'))
        flag = True
        pos_pairs_cos1 = 0
        neg_pairs_cos1 = 0
        loss1 = 0
        for A, B in zip(featureA, featureB):
            
            A = torch.matmul(A, A.t())
            B = torch.matmul(B, B.t())
            
            A_pos = F.normalize(A, p=2, dim=1)  # (-1, 1)
            dot_prod = F.normalize(B, p=2, dim=1)
            A_neg = (0 - A_pos).float()

            pos_pairs_cos = torch.cosine_similarity(A_pos, dot_prod, dim=1).mean()
            neg_pairs_cos = torch.cosine_similarity(A_neg, dot_prod, dim=1).mean()
            
            loss = (1.0 - pos_pairs_cos) + torch.abs(self.gamma * neg_pairs_cos)
            pos_pairs_cos1 = pos_pairs_cos
            neg_pairs_cos1 = neg_pairs_cos
            loss1 = loss
            total_loss += loss

        return total_loss / 64, pos_pairs_cos1, neg_pairs_cos1
