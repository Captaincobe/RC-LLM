import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
def KL(alpha, c):
    beta = torch.ones((1, c)).to(device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def get_dc_loss(alphas, device):
    """
    输入：
        alphas: List[Tensor]，长度为 num_views, 每个 Tensor 形状为 (batch_size, num_classes)
                表示每个视角的 Dirichlet 参数 alpha
        device: torch.device, 用于 Tensor 分配

    输出：
        dc_loss: 冲突损失值(scalar),用于训练中最小化
    """
    num_views = len(alphas)
    batch_size, num_classes = alphas[0].shape

    p = torch.zeros((num_views, batch_size, num_classes), device=device)  # 每个视角的期望分布
    u = torch.zeros((num_views, batch_size), device=device)               # 每个视角的不确定度

    for v in range(num_views):
        alpha = alphas[v]  # 直接使用输入的 α
        S = torch.sum(alpha, dim=1, keepdim=True)       # S = ∑ α_i
        p[v] = alpha / S                                # belief expectation
        u[v] = (num_classes / S).view(-1)               # uncertainty u = K / S

    dc_sum = 0
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            pd = torch.sum(torch.abs(p[i] - p[j]) / 2, dim=1)  # probability disagreement
            cc = (1 - u[i]) * (1 - u[j])                       # confidence
            dc_sum += torch.sum(pd * cc)
            count += batch_size

    dc_loss = dc_sum / count
    return dc_loss


def get_soft_dc_loss(alphas, device, beta=1.0):
    """
    软置信度版本的视角冲突损失。
    """
    num_views = len(alphas)
    batch_size, num_classes = alphas[0].shape

    p = torch.zeros((num_views, batch_size, num_classes), device=device)
    u = torch.zeros((num_views, batch_size), device=device)

    for v in range(num_views):
        alpha = alphas[v]
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = num_classes / S.view(-1)

    dc_sum = 0
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            # KL散度 (更平滑)
            kl_ij = F.kl_div(p[i].log(), p[j], reduction='none').sum(dim=1)
            kl_ji = F.kl_div(p[j].log(), p[i], reduction='none').sum(dim=1)
            kl = (kl_ij + kl_ji) / 2  # D_KL

            # Soft Confidence
            # 方式A：乘积开根号
            conf = torch.sqrt((1 - u[i]) * (1 - u[j]))  # shape = (B,)
            # 或 方式B：sigmoid权重
            # conf = torch.sigmoid(beta * ((1 - u[i]) + (1 - u[j])) / 2)

            # 融合
            dc_sum += torch.sum(kl * conf)
            count += batch_size

    return dc_sum / count


# def get_soft_dc_loss(alphas, device, beta=1.0):
#     """
#     软置信度版本的视角冲突损失。
#     """
#     num_views = len(alphas)
#     batch_size, num_classes = alphas[0].shape

#     p = torch.zeros((num_views, batch_size, num_classes), device=device)
#     u = torch.zeros((num_views, batch_size), device=device)

#     # 1. 计算每个 view (agent) 的期望概率 p 和不确定性 u
#     for v in range(num_views):
#         alpha = alphas[v]
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         p[v] = alpha / S  # 这就是论文中的 \hat{p}^s
#         u[v] = num_classes / S.view(-1) # 这就是论文中的 u^s

#     dc_sum = 0
#     count = 0 # count 变量似乎没有在最终返回的 loss 中使用，而是直接返回 dc_sum
#               # 如果要计算平均损失，应该用 dc_sum / count
#               # 但通常这种 loss component 是 sum over batch，然后在总 loss 中由 batch_size 平均

#     for i in range(num_views):
#         for j in range(i + 1, num_views): # 两两 view 之间计算
#             # 2. 计算对称 KL 散度
#             # KL散度 (更平滑)
#             # F.kl_div(input, target, ...) expects input to be log-probabilities and target to be probabilities
#             kl_ij = F.kl_div(p[i].log(), p[j], reduction='none').sum(dim=1) # D_KL(p[j] || p[i]) if p[i] is log
#                                                                         # Actually, F.kl_div(x, y) = sum(y * (log(y) - x)) if x is log_prob
#                                                                         # So, F.kl_div(p[i].log(), p[j]) = sum(p[j] * (log(p[j]) - p[i].log()))
#                                                                         # This is D_KL(p[j] || p[i])
#             kl_ji = F.kl_div(p[j].log(), p[i], reduction='none').sum(dim=1) # D_KL(p[i] || p[j])
#             kl = (kl_ij + kl_ji) / 2 # 对称 KL 散度

#             # 3. Soft Confidence (置信度对齐)
#             # 方式A：乘积开根号
#             conf = torch.sqrt((1 - u[i]) * (1 - u[j]))  # shape = (B,)
#             # 或 方式B：sigmoid权重 (未使用)
#             # conf = torch.sigmoid(beta * ((1 - u[i]) + (1 - u[j])) / 2)

#             # 4. 融合 (计算带权重的冲突)
#             dc_sum += torch.sum(kl * conf) # 对 batch 内所有样本的 (kl * conf)求和，并累加到 dc_sum
#             # count += batch_size # 如果要算平均值，分母应该是样本对的数量乘以 batch_size，或者直接返回 sum

#     # 函数的最后没有显式 return 语句，但从下面的 loss 计算来看，
#     # get_soft_dc_loss(...) 应该返回 dc_sum (或 dc_sum / count 如果要做平均)
#     # 假设它返回的是 dc_sum
#     return dc_sum # 假设是这样，或者是一个平均值


# def get_dc_loss(alphas, device, visualize=True):
#     num_views = len(alphas)
#     batch_size, num_classes = alphas[0].shape
#     p = torch.zeros((num_views, batch_size, num_classes), device=device)
#     u = torch.zeros((num_views, batch_size), device=device)

#     for v in range(num_views):
#         alpha = alphas[v]
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         p[v] = alpha / S
#         u[v] = (num_classes / S).view(-1)

#     conf = torch.softmax(-u, dim=0)
#     dc_sum = 0.0
#     pair_count = 0
#     conflicts = []

#     for i in range(num_views):
#         for j in range(i + 1, num_views):
#             pd = torch.sum(torch.abs(p[i] - p[j]) / 2, dim=1)
#             weight = conf[i] * conf[j]
#             dc = pd * weight
#             conflicts.append(dc.detach().cpu())
#             dc_sum += torch.sum(dc)
#             pair_count += batch_size

#     if visualize:
#         avg_conflict = torch.stack(conflicts).mean(dim=1).numpy()
#         plt.figure(figsize=(6, 3))
#         plt.bar(range(len(avg_conflict)), avg_conflict)
#         plt.title("Average Conflict Score per View Pair")
#         plt.xlabel("View Pair Index")
#         plt.ylabel("Conflict")
#         plt.tight_layout()
#         plt.savefig("conflict_scores.png")
#         plt.close()

#     dc_loss = dc_sum / pair_count
#     return dc_loss


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)          # Dirichlet参数的总和
    E = alpha - 1                                       # 代表 evidence
    label = F.one_hot(p.long(), num_classes=c)         # one-hot 标签向量

    # 监督CE loss部分（Evidential CrossEntropy）
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # KL 散度部分：正则项
    annealing_coef = min(1, global_step / annealing_step)    # 动态调节KL权重
    alp = E * (1 - label) + 1                                # 对非目标类的evidence保留（推开）
    B = annealing_coef * KL(alp, c)

    return (A + B)


