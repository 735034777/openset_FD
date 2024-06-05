from src.train.CWRU.result.find_max_youdens_records import find_max_youdens_records


file_path = r"H:\project\imbalanced_data\openset_FD\src\train\NPP_PCTRAN\result_process\SDOS_SELECTED_COMBINATION.csv"
max_youdens_index_records = find_max_youdens_records(file_path)
max_youdens_index_records.to_csv("SODS_MAX_YOUDENS.csv")
# print()

