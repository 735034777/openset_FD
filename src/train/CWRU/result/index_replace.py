

import pandas as pd
import os

# Load your data
filename = "index_CWRU.csv"
file_path = r"H:\project\imbalanced_data\openset_FD\src\train\CWRU"
sorce_file_path = os.path.join(file_path,filename)
df = pd.read_csv(sorce_file_path)

# Define your labels and their corresponding indices
all_labels = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007@6', 'OR014@6', 'OR021@6']
# label_to_index = {label: index for index, label in enumerate(all_labels)}


import pandas as pd
import ast  # 用于安全地将字符串形式的列表转换为列表

# 假设`df`是已经加载的DataFrame
# all_labels定义如前所述

all_labels = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007@6', 'OR014@6', 'OR021@6',"normal"]
label_to_index = {label: index for index, label in enumerate(all_labels)}

# 函数：将字符串形式的列表转换为索引列表
def convert_fault_types_to_indices(fault_types_str):
    # 安全地将字符串转换为列表
    fault_types_list = ast.literal_eval(fault_types_str)
    # 将故障类型转换为索引
    indices_list = ["c"+str(label_to_index[ftype]+1) for ftype in fault_types_list]
    # 返回转换后的索引列表
    return indices_list

# 应用转换函数
# 假设故障类型列的名称是'Fault Types'
df['TRAIN_LABEL'] = df['TRAIN_LABEL'].apply(convert_fault_types_to_indices)
df['TEST_LABEL'] = df['TEST_LABEL'].apply(convert_fault_types_to_indices)
# df['open fault'] = df['open fault'].apply(convert_fault_types_to_indices)
df.to_csv("modify_"+filename)

# 查看转换后的DataFrame
print(df.head())
