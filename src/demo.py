import matplotlib.pyplot as plt

# 定义第一个函数，用于生成第一个图
def plot_first_figure():
    x = [1, 2, 3, 4]
    y = [10, 20, 25, 30]
    plt.plot(x, y, color='blue')
    plt.title("First Plot")

# 定义第二个函数，用于生成第二个图
def plot_second_figure():
    x = [1, 2, 3, 4]
    y = [40, 30, 20, 10]
    plt.plot(x, y, color='red')
    plt.title("Second Plot")

# 创建一个母图
plt.figure()

# 在母图上添加第一个子图
plt.subplot(1, 2, 1)  # 1行2列的第1个位置
plot_first_figure()

# 在母图上添加第二个子图
plt.subplot(1, 2, 2)  # 1行2列的第2个位置
plot_second_figure()

# 显示最终的母图
plt.show()
