import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from gensim.models import LsiModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np

import gensim.utils
from gensim import corpora
from pprint import pprint
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models

from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import nltk

from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import output_file

from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
import pickle

x = open("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('//Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)



# create LSI model
lsimodel = LsiModel(corpus=corpus, num_topics=13, id2word=id2word)

# print topics learned by LSI model
pprint(lsimodel.print_topics())

# create an empty list to store topic weights for each document
topic_weights = []

# iterate through each document in the corpus
for i, row_list in enumerate(lsimodel[corpus]):
    # for each document, append its topic weights to the topic_weights list
    topic_weights.append([w for i, w in row_list])

# convert the list of topic weights to a numpy array
arr = pd.DataFrame(topic_weights).fillna(0).values

# keep only the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# determine the dominant topic for each document
topic_num = np.argmax(arr, axis=1)

# use t-SNE for dimensionality reduction to visualize dominant topics in 2D space
tsne_model = TSNE(n_components=2, verbose=1, random_state=42, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# print the result of t-SNE dimensionality reduction
pprint(tsne_lda)



#
#
# df_tsne = pd.DataFrame(tsne_lda, columns=['x', 'y'])
# df_tsne['dominant_topic'] = topic_num
# df_tsne['document'] = [str(doc) for doc in final_data[:len(df_tsne)]]
#
#
#
#
# # Create a ColumnDataSource from the DataFrame
# source = ColumnDataSource(df_tsne)
#
# # List of colors to use for indicating topics
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF', '#ff0343', '#9c2dff']
#
# # Create the figure
# p = figure(width=1500, height=900,
#            tools="pan, wheel_zoom, box_zoom, reset, hover, save",
#            x_axis_type=None, y_axis_type=None, min_border=1)
#
# # Add a hover tool to display information
# hover = HoverTool(mode='mouse')
# hover.tooltips = [("Topic", "@dominant_topic"), ("Document", "@document")]
# hover.tooltips.border_line_width = 3
# hover.tooltips.max_width = 400  # Sets the max width to 400px
#
# p.add_tools(hover)
#
# # Draw the points on the plot
# for i, color in enumerate(colors):
#     df_subset = df_tsne[df_tsne.dominant_topic == i]
#     source_subset = ColumnDataSource(df_subset)
#     p.circle(x='x', y='y', legend_label=f"Topic {i}", source=source_subset, color=color, alpha=0.6, size=5)
#
# # Configure plot attributes
# p.legend.location = "bottom_left"
# p.legend.click_policy = "hide"
# p.legend.title= "Dominant Topics"
#
# # Specify the path for saving
# output_file("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_tSNE_interactive.html")
#
# # show(p)
# save(p) # Save the plot without displaying it


