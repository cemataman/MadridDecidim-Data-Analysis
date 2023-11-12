import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# Read the CSV file into a DataFrame
df_dominant_topic = pd.read_csv("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_dominant_topic.csv")


### Aggregate by topic -------------------------------------------------------------------------------------
topic_aggregation = df_dominant_topic.groupby('Dominant_Topic')['Topic_Perc_Contrib'].mean()

# Generate colors from the 'viridis' color map
colors = plt.cm.viridis(np.linspace(0, 1, len(topic_aggregation)))

# Create bar plot with custom colors
plt.bar(topic_aggregation.index, topic_aggregation.values, color=colors)
plt.xlabel('Topic')
plt.ylabel('Average Topic Percentage Contribution')
plt.title('Average Topic Contribution Across All Documents')

# Save the plot as a PNG file
plt.savefig('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_Average_Topic_Contribution.png')
plt.show()



### Create density plot-------------------------------------------------------------------------------------
sns.kdeplot(data=df_dominant_topic, x='Topic_Perc_Contrib', hue='Dominant_Topic', palette='viridis')
plt.title('Density Plot of Topic Contributions')
plt.xlabel('Topic Percentage Contribution')
plt.ylabel('Density')

# Save the plot as a PNG file
plt.savefig('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_Density_Plot_Topic_Contributions.png')
plt.show()


### Create bins------------------------------------------------------------------------------------------------
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
df_dominant_topic['Contrib_Bin'] = pd.cut(df_dominant_topic['Topic_Perc_Contrib'], bins=bins)

# Count the number of documents in each bin for each topic
bin_counts = df_dominant_topic.groupby(['Dominant_Topic', 'Contrib_Bin']).size().reset_index(name='Counts')

# Create bar plot with custom axis labels and color scheme
sns.barplot(x='Contrib_Bin', y='Counts', hue='Dominant_Topic', data=bin_counts, palette='viridis')
plt.title('Number of Documents in Each Contribution Bin')
plt.xlabel('Contribution Bins')
plt.ylabel('Number of Documents')
# Save the plot as a PNG file
plt.savefig('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_Number_of_Documents_in_Contribution_Bins.png')
plt.show()
