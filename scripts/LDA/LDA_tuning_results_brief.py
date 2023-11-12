import pandas as pd
import os

# Read the CSV file into a pandas dataframe
df = pd.read_csv('/data/intermediate_data/lda_tuning_results.csv')

# Group the dataframe by the 'a' column (topic names) and find the max value in column 'c'
max_values = df.groupby('Topics').agg({'Coherence': 'max'}).reset_index()

# Merge the max_values dataframe with the original dataframe to get the corresponding values in columns
result = pd.merge(max_values, df, on=['Topics', 'Coherence'], how='left')[['Topics', 'Alpha', 'Beta', 'Coherence', 'Perplexity']].reset_index(drop = True)

# Save the merged dataframe as a new CSV file
output_path = os.path.join('/data/intermediate_data', 'lda_tuning_results_brief.csv')
result.to_csv(output_path, index=False)
