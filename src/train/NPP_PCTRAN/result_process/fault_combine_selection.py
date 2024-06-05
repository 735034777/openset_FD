from src.train.CWRU.result.fault_combine_selection import process_experiment


file_path = r"H:\project\imbalanced_data\openset_FD\src\train\NPP_PCTRAN\SDOS_RESULTS_DIFF_METRIC.csv"
result = process_experiment(file_path,"youdens_index",0.7)
result.to_csv("SDOS_SELECTED_COMBINATION.csv")