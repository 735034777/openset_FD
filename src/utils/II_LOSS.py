import torch
import torch.nn.functional as F

def ii_loss(outputs, labels, num_classes):
    # 初始化类均值列表
    class_means = []
    for i in range(num_classes):
        class_members = outputs[labels == i]
        if len(class_members) > 0:
            class_means.append(class_members.mean(dim=0))
        else:
            class_means.append(torch.zeros_like(outputs[0]))

    class_means = torch.stack(class_means)

    # 计算类内扩散
    intra_spread = 0
    for i in range(num_classes):
        class_members = outputs[labels == i]
        if len(class_members) > 0:
            intra_spread += F.mse_loss(class_members, class_means[i].expand_as(class_members))

    intra_spread /= num_classes

    # 计算类间分离
    inter_separation = torch.max(torch.cdist(class_means, class_means, p=2))

    # ii-Loss
    loss = -1* inter_separation+intra_spread
    return loss

if __name__=="__main__":
    y = torch.random
