import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tomotopy as tp
import numpy as np
from pprint import pprint
from gensim.models import CoherenceModel
from gensim import corpora
import csv
import pickle
import matplotlib.pyplot as plt
import os

# Define a function to extract the most salient topics and their associated words from an HDP model.
def get_hdp_topics(hdp, top_n=5):
    # Sort topics by frequency of assignment, with the most frequent topics appearing first.
    sorted_topics = [k for k, v in sorted(enumerate(hdp.get_count_by_topics()), key=lambda x: x[1], reverse=True)]

    topics = dict()

    # Iterate through the sorted topics and filter out topics that are not 'alive' (i.e., unassigned).
    for k in sorted_topics:
        if not hdp.is_live_topic(k): continue
        topic_wp = []
        # Extract the top 'n' words for each topic based on their probabilities.
        for word, prob in hdp.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp  # Store the words and their frequencies in relation to the topic.

    return topics


# Load data from an external Python file.
x = open("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Initialize the Hierarchical Dirichlet Process (HDP) model with specific hyperparameters.
term_weight = tp.TermWeight.ONE
hdp = tp.HDPModel(tw=term_weight, min_cf=10, rm_top=15, gamma=1, alpha=2, initial_k=1, seed=9999)

# Add documents to the HDP model for training.
for vec in final_data:
    hdp.add_doc(vec)

### HYPERPARAMETER TUNING ------------------------------------------------------------------------

# Specify the number of initial iterations to discard for burn-in.
hdp.burn_in = 100
hdp.train(0)
print('Num docs:', len(hdp.docs), ', Vocab size:', hdp.num_vocabs, ', Num words:', hdp.num_words)
print('Removed top words:', hdp.removed_top_words)

# Execute the training of the HDP model through Markov Chain Monte Carlo (MCMC) iterations.
mcmc_iter = 1000
for i in range(0, mcmc_iter, 100):
    hdp.train(50, workers=3)
    print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, hdp.ll_per_word, hdp.live_k))

# Extract the trained topics and their constituent words.
topics = get_hdp_topics(hdp)
pprint(topics)

# Test the trained HDP model on a sample document.
test_doc = final_data[0]
doc_inst = hdp.make_doc(test_doc)
topic_dist, ll = hdp.infer(doc_inst)
topic_idx = np.array(topic_dist).argmax()
pprint(topic_idx)

# Retrieve the words that are most representative of the chosen topic.
x = hdp.get_topic_words(topic_idx)
pprint(x)

from wordcloud import WordCloud


def plot_wordcloud_freq(topics, min_freq=0.02, save_path='/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/hdp_wordclouds'):
    for topic_id, words in topics.items():
        wc = WordCloud(background_color='white', width=800, height=400)
        word_freq = {word: freq for word, freq in words if freq >= min_freq}

        if len(word_freq) == 0:
            print(f"Topic {topic_id} has no words above the frequency threshold.")
            continue

        wc.generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id}')

        # Save the figure
        plt.savefig(os.path.join(save_path, f'freq_wordcloud_topic_{topic_id}.png'))
        plt.show()


def plot_wordcloud_top_n(topics, top_n=50, save_path='/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/hdp_wordclouds'):
    for topic_id, words in topics.items():
        wc = WordCloud(background_color='white', width=800, height=400)
        word_freq = {word: freq for word, freq in words}

        # Sort by frequency and keep top_n words
        word_freq = {k: word_freq[k] for k in sorted(word_freq, key=word_freq.get, reverse=True)[:top_n]}

        wc.generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id}')

        # Save the figure
        plt.savefig(os.path.join(save_path, f'top_n_wordcloud_topic_{topic_id}.png'))
        plt.show()

plot_wordcloud_freq(topics)
plot_wordcloud_top_n(topics)







