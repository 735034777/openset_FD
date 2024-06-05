import pandas as pd


def process_experiment(file_path, max_value_column, threshold):
    # 读取数据
    data = pd.read_csv(file_path)

    # 准备一个空的DataFrame来保存结果
    result_df = pd.DataFrame()
    ExperimentID = 1

    # 分割数据，每40行为一个实验
    for start_row in range(0, len(data), 40):
        experiment_data = data.iloc[start_row:start_row + 40]

        # 存储每个变量组的最大值
        max_values = []

        # 分割实验数据，每10行为一个变量组
        for var_start_row in range(0, 40, 10):
            variable_data = experiment_data.iloc[var_start_row:var_start_row + 10]

            # 找到指定列的最大值，并添加到列表中
            max_value = variable_data[max_value_column].max()
            max_values.append(max_value)
        # 示例列表
        list_with_nan = max_values
        series = pd.Series(list_with_nan)

        # 判断是否存在NaN值
        has_nan = series.isna().any()
        has_nan = False

        # 判断四个最大值中的最小值是否大于阈值
        if min(max_values) > threshold and not has_nan:
            # 如果条件满足，将这40行数据保存到结果DataFrame中
            experiment_data['ExperimentID'] = ExperimentID  # 创建实验号
            ExperimentID = ExperimentID+1
            result_df = pd.concat([result_df, experiment_data], ignore_index=True)

    return result_df

if __name__=="__main__":
    file_path = r"H:\project\imbalanced_data\openset_FD\src\train\CWRU\SDOS_RESULTS_DIFF_METRIC.csv"
    result_df = process_experiment(file_path,"youdens_index",0.7)
    result_df.to_csv("SDOS_RESULTS.csv")