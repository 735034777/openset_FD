class PctranData:
    def __init__(self,foldpath,powerdemand="100",maxtimestep = 2500):
        # 定义 foldpath 变量，用来存储文件路径
        self.foldpath = foldpath
        # 定义 powerdemand 变量，用来存储功率需求
        self.powerdemand = powerdemand
        # 定义 maxtimestep 变量，用来存储最大时间步长
        self.maxtimestep = maxtimestep
