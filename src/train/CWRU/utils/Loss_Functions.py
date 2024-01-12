import torch.nn as nn
import torch.nn.functional as F
import torch
def get_crossentropyloss():
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_tripletloss(ohem_rate=0.5):
    def criterion(logic,target):
        loss = criterion(logic, target)
        # OHEM：选择损失最大的一部分样本
        ohem_loss, _ = torch.topk(loss, int(ohem_rate * loss.size(0)))
        # 困难样本的平均损失
        ohem_loss = ohem_loss.mean()
        Loss = nn.TripletMarginLoss(margin=1.0, p=2)
    return criterion()

def get_ii_loss():
    def ii_loss(logic,target):
        class_means = []
        num_classes = len(torch.unique(target))
        for i in range(num_classes):
            class_members = logic[target == i]
            if len(class_members) > 0:
                class_means.append(class_members.mean(dim=0))
            else:
                class_means.append(torch.zeros_like(logic[0]))

        class_means = torch.stack(class_means)

        # 计算类内扩散
        intra_spread = 0
        for i in range(num_classes):
            class_members = logic[target == i]
            if len(class_members) > 0:
                intra_spread += F.mse_loss(class_members, class_means[i].expand_as(class_members))

        intra_spread /= num_classes

        # 计算类间分离
        inter_separation = torch.max(torch.cdist(class_means, class_means, p=2))

        # ii-Loss
        loss = -1* inter_separation+intra_spread
        return loss
    return ii_loss


