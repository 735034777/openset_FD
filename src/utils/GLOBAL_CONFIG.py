from src.utils.CNN_LSTM import CustomModel
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import wandb



class GLOABAL_CONFIG:
    def __init__(self):
        self.epoch = 300  # 训练的轮数
        self.source_file_path = ""  # 源文件路径
        self.model_save_path = ""  # 模型保存路径
        self.experiment_dataset = "CWRU"  # 实验数据集
        self.experiment_title = "tailsize"  # 实验标题
        self.randomseed = 42  # 随机数种子
        self.lr = 1e-3  # 学习率
        self.processed_file_path = ""  # 处理后的文件路径
        self.dim = 4  # 维度
        self.metric = "cosine"  # 距离度量
        self.tailsize = 5  # 尾大小
        self.lr_delay_method = "cosine"  # lr延迟方法
        self.experiment_log_method_name = "wandb"  # 实验日志方法名


    def set_base_model(self):
        self.base_model = CustomModel(self.dim)  # 设置基础模型
        self.modelname = "CNN_LSTM"  # 模型名称

    def set_loss_function(self):
        self.loss_function = cross_entropy()  # 设置损失函数
        self.loss_function_name = "Cross_entropy"  # 损失函数名称

    def set_optimizer(self):
        self.optimizer = Adam(lr=self.lr)  # 设置优化器
        self.optimizer_name = "Adam"  # 优化器名称

    def set_wandb_config(self):
        wandb.init(  # wandb配置
            project="openmax"+"_"+self.experiment_dataset,
            config = {
                "randomseed":self.randomseed,
                "tailsize":self.tailsize,
                "metric":self.metric,
                "modelname":self.modelname
            }
        )
