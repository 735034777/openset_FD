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
    loss = intra_spread - inter_separation
    return loss


# 测试案例
def test_ii_loss():
    # 生成模拟数据
    num_samples = 10
    num_features = 5
    num_classes = 3

    # 随机生成outputs和labels
    outputs = torch.randn(num_samples, num_features)
    labels = torch.randint(0, num_classes, (num_samples,))

    # 调用函数计算损失
    loss = ii_loss(outputs, labels, num_classes)

    # 检查loss是否为一个标量张量
    assert loss.dim() == 0, "Loss is not a scalar value."

    # 打印loss用于验证
    print(f"Computed loss: {loss.item()}")




if __name__=="__main__":
    test_ii_loss()
