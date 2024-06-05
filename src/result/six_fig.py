import matplotlib.pyplot as plt
import numpy as np
from cwru_recall import plot_cwru_recall
from cwru_f1 import plot_cwru_f1
from cwru_youdenindex import plot_cwru_youden
from pctran_recall import plot_pctran_recall
from pctran_f1 import plot_pctran_f1
from pctran_youdenindex import plot_pctran_youdenindex

# Setting up the figure for multiple subplots
fig, axs = plt.subplots(3, 2, figsize=(16, 18))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.1)
plot_cwru_recall(axs[0,0])
plot_cwru_f1(axs[0,1])
plot_cwru_youden(axs[1,0])
plot_pctran_recall(axs[1,1])
plot_pctran_f1(axs[2,0])
plot_pctran_youdenindex(axs[2,1])


# Simulate data for demonstration
# np.random.seed(0)  # For consistent random data
# x = np.linspace(0, 0.4, 5)
#
# # Iterate over each subplot in a 3x2 grid
# for ax_row in axs:
#     for ax in ax_row:
#         # for i in range(12):  # Each subplot will contain 12 lines
#             # y = np.random.rand(5) * (1 - i * 0.05) + 0.5  # Simulated Y values for each line
#             # ax.plot(x, y, label=f'Line {i+1}', marker='o', linestyle='-')
#         # fig, axs = plt.subplots(3, 2, figsize=(16, 18))
#         # Use the function in one of the subplots
#         # plot_cwru_recall(axs[0, 0])
#         # plot_cwru_recall([])
# plot_cwru_recall(ax)
# ax.set_title('Simulated Dataset')
# ax.set_xlabel('X Axis (0 to 1)')
# ax.set_ylabel('Y Axis (0 to 1)')
# ax.grid(True)
# ax.legend(fontsize='small', ncol=2)

plt.tight_layout()
plt.show()
# 或者保存为SVG格式
plt.savefig('result_plot.svg', format='svg')
