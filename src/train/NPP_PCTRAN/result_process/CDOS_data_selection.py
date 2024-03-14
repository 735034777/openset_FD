#筛选CDOS_PCTRAN实验中表现较好的故障组合

import pandas as pd
from src.config import BASE_FILE_PATH
df = pd.read_csv(BASE_FILE_PATH+r"\src\train\NPP_PCTRAN\CDOS_RESULTS_DIFF_METRIC.csv")

# Since directly grouping by columns with complex data types like lists and sets is difficult in pandas,
# we will focus on using 'index_c' as a proxy for grouping, assuming it uniquely identifies groups formed
# by 'TRAIN_LABEL', 'TEST_LABEL', and 'open fault' for the simplicity of demonstration.

# Step 1: Find the maximum TEST_ACCURACY within each group identified by 'index_c'
max_test_accuracy_by_group = df.groupby('index_c')['TEST_ACCURACY'].transform('max')

# Step 2: Add this maximum TEST_ACCURACY to the original dataframe as a new column for sorting purposes
df['MAX_TEST_ACCURACY_BY_GROUP'] = max_test_accuracy_by_group

# Step 3: Sort the dataframe by this new column to ensure that rows are sorted according to the maximum TEST_ACCURACY within their group
df_sorted_by_max_test_accuracy = df.sort_values(by=['MAX_TEST_ACCURACY_BY_GROUP', 'index_c'], ascending=[False, True])

# Display the first few rows of the sorted dataframe to verify the sorting logic
df_sorted_by_max_test_accuracy[['index_c', 'TRAIN_LABEL', 'TEST_LABEL', 'open fault', 'TEST_ACCURACY', 'MAX_TEST_ACCURACY_BY_GROUP']].head()

print()

