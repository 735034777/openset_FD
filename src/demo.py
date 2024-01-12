import matplotlib.pyplot as plt

# �����һ���������������ɵ�һ��ͼ
def plot_first_figure():
    x = [1, 2, 3, 4]
    y = [10, 20, 25, 30]
    plt.plot(x, y, color='blue')
    plt.title("First Plot")

# ����ڶ����������������ɵڶ���ͼ
def plot_second_figure():
    x = [1, 2, 3, 4]
    y = [40, 30, 20, 10]
    plt.plot(x, y, color='red')
    plt.title("Second Plot")

# ����һ��ĸͼ
plt.figure()

# ��ĸͼ����ӵ�һ����ͼ
plt.subplot(1, 2, 1)  # 1��2�еĵ�1��λ��
plot_first_figure()

# ��ĸͼ����ӵڶ�����ͼ
plt.subplot(1, 2, 2)  # 1��2�еĵ�2��λ��
plot_second_figure()

# ��ʾ���յ�ĸͼ
plt.show()
