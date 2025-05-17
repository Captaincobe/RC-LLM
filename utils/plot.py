import numpy as np
import matplotlib.pyplot as plt

# ========================
# Matplotlib 风格配置（模仿目标图）
# ========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# # ========================
# # 数据读取
# # ========================
# dataset = "TONIoT"
# dc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4]
# result_file = f"results/{dataset}/summary.txt"

# # summary.txt 每行: Accuracy, Precision, Recall, F1-score
# data = np.loadtxt(result_file)
# assert data.shape[0] == len(dc_values), "数据长度与 dc_loss 数量不一致"

# # 转为百分比
# data = data * 100

# ========================
# 读取多个数据集的 F1-score
# ========================
datasets = ["CICIDS", "DoHBrw", "TONIoT"]
labels = ["CICIDS", "DoHBrw", "TONIoT"]
colors = ["firebrick", "royalblue", "darkgreen"]
dc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4]

f1_scores = []
for ds in datasets:
    result_file = f"results/{ds}/summary.txt"
    data = np.loadtxt(result_file)
    assert data.shape[0] == len(dc_values), f"{ds} 数据长度与 dc_loss 数量不一致"
    f1 = data[:, 3] * 100  # F1-score 是第 4 列
    f1_scores.append(f1)

# ========================
# 绘制 F1-score 曲线图（每条线代表一个数据集）
# ========================
# 创建等距位置
x_pos = np.arange(len(dc_values))
x_labels = [str(v) for v in dc_values]

plt.figure(figsize=(4.5, 4))
# plt.figure(figsize=(6.5, 4))
for f1, label, color in zip(f1_scores, labels, colors):
    plt.plot(x_pos, f1, marker='o', color=color, label=label, linewidth=2)

plt.xlabel(r"$\lambda_c$")
plt.ylabel("F1-score (%)")
plt.xticks(x_pos, x_labels)  # 把刻度设为等距位置，对应的标签为字符串
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/conf.pdf", dpi=600)
plt.show()



# # ========================
# # 断轴绘图（F1-score）
# # ========================
# x_pos = np.arange(len(dc_values))
# x_labels = [str(v) for v in dc_values]

# # 创建上下两个子图共享 x 轴
# fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(4.5, 3.8), gridspec_kw={'height_ratios': [2, 1]})

# # 设置 y 轴显示范围（跳过中间空白）
# ax_top.set_ylim(94, 97)       # 上部分显示高 F1
# ax_bot.set_ylim(86.5, 89)     # 下部分显示低 F1

# # 绘图：分别画到两个 subplot 上
# for f1, label, color in zip(f1_scores, labels, colors):
#     ax_top.plot(x_pos, f1, marker='o', color=color, label=label, linewidth=2)
#     ax_bot.plot(x_pos, f1, marker='o', color=color, linewidth=2)
# # 断轴处理：隐藏边框线 + 添加锯齿符号
# ax_top.spines['bottom'].set_visible(False)
# ax_bot.spines['top'].set_visible(False)
# # ax_top.tick_params(labeltop=False)
# ax_top.tick_params(axis='x', which='both', bottom=False, top=False)
# ax_bot.xaxis.tick_bottom()

# # 添加视觉断裂符号（“锯齿”）
# # kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8,
# #               linestyle='none', color='k', mec='k', mew=1)
# # ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
# # ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

# # 坐标轴设置
# plt.xticks(x_pos, x_labels)
# ax_bot.set_xlabel(r"$\lambda_c$", fontsize=13)
# ax_top.set_ylabel("F1-score (%)", fontsize=13)
# ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=3, frameon=False)
# ax_top.grid(axis='y', linestyle='--', alpha=0.6)
# ax_bot.grid(axis='y', linestyle='--', alpha=0.6)
# # ax_top.grid(True, linestyle='--', alpha=0.6)
# # plt.grid(True, linestyle='--', alpha=0.6)
# plt.subplots_adjust(hspace=0.05) 
# plt.tight_layout()
# plt.savefig("results/f1score.pdf", dpi=600)
# plt.show()
