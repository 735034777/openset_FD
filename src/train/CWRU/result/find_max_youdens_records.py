import pandas as pd

def find_max_youdens_records(data_path):
    """
    Load data from the given path, find the record with the maximum Youden's index for each combination of
    ExperimentID and METRIC_TYPE, and return a DataFrame containing these records.

    Parameters:
    - data_path: str, the file path of the source data.

    Returns:
    - DataFrame with records having the maximum Youden's index for each ExperimentID and METRIC_TYPE combination.
    """
    # Load the data
    data = pd.read_csv(data_path)

    # Initialize an empty dataframe to collect the correct records
    max_youdens_index_records = pd.DataFrame()

    # Loop over each combination of ExperimentID and METRIC_TYPE, filtering out NaN values for youdens_index first
    for (experiment_id, metric_type), group in data.groupby(['ExperimentID', 'METRIC_TYPE']):
        # Filter out NaN values for youdens_index
        group_nonan = group.dropna(subset=['youdens_index'])
        # If the group is not empty after filtering, find the record with the max Youden's index
        if not group_nonan.empty:
            max_record = group_nonan.loc[group_nonan['youdens_index'].idxmax()]
            # Append this record to the collection dataframe
            max_youdens_index_records = pd.concat([max_youdens_index_records, pd.DataFrame([max_record])],
                                                  ignore_index=True)

    return max_youdens_index_records


if __name__ =="__main__":
    # Test the function with the previously used data path as an example
    test_data_path = r'H:\project\imbalanced_data\openset_FD\src\train\CWRU\result\SDOS_RESULTS.csv'
    test_result = find_max_youdens_records(test_data_path)
    # test_result.head()
    test_result.to_csv("SDOS_max_youdens.csv")
