import tomotopy as tp
from gensim.models import CoherenceModel
from gensim import corpora
import pickle

# Define a function to extract the most important topics and their corresponding words from a HDP model
def get_hdp_topics(hdp, top_n=10):
    # Get most important topics by # of times they were assigned (i.e. counts)
    sorted_topics = [k for k, v in sorted(enumerate(hdp.get_count_by_topics()), key=lambda x: x[1], reverse=True)]

    topics = dict()

    # For topics found, extract only those that are still assigned
    for k in sorted_topics:
        if not hdp.is_live_topic(k): continue  # remove un-assigned topics at the end (i.e. not alive)
        topic_wp = []
        for word, prob in hdp.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp  # store topic word/frequency array
    return topics


# Read in data from a file
x = open("/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

from sklearn.model_selection import KFold

# Initialize an HDP model with some parameters
term_weight = tp.TermWeight.ONE
hdp = tp.HDPModel(tw=term_weight, min_cf=10, rm_top=15, gamma=1, alpha=2, initial_k=1, seed=42)

# Add documents to the model
for vec in final_data:
    hdp.add_doc(vec)

# Set burn-in iterations and train the model
hdp.burn_in = 100
hdp.train(0)
print('Num docs:', len(hdp.docs), ', Vocab size:', hdp.num_vocabs, ', Num words:', hdp.num_words)
print('Removed top words:', hdp.removed_top_words)

# Train the model with MCMC sampling
mcmc_iter = 1000
hdp.train(mcmc_iter, workers=3)

# Extract the topics and their words from the trained model
topics = get_hdp_topics(hdp)
topic_terms = [[term for term, _ in topics[topic]] for topic in topics]

# Initialize KFold cross-validation
kf = KFold(n_splits=10)

# Create an empty list to store the coherence scores
coherences = []

# Perform 10-fold cross-validation
for train_idx, test_idx in kf.split(final_data):
    # Create a new HDP model for each fold
    fold_hdp = tp.HDPModel(tw=term_weight, min_cf=10, rm_top=15, gamma=1, alpha=2, initial_k=1, seed=42)

    # Add documents to the model for the training set
    for idx in train_idx:
        fold_hdp.add_doc(final_data[idx])

    # Train the model with MCMC sampling
    fold_hdp.burn_in = 100
    fold_hdp.train(0)
    fold_hdp.train(mcmc_iter, workers=3)

    # Extract the topics and their words from the trained model
    fold_topics = get_hdp_topics(fold_hdp)
    fold_topic_terms = [[term for term, _ in fold_topics[topic]] for topic in fold_topics]

    # Calculate coherence score using Gensim's CoherenceModel for the test set
    cm = CoherenceModel(topics=fold_topic_terms, texts=[final_data[idx] for idx in test_idx], dictionary=id2word, coherence='u_mass')
    coherence_score = cm.get_coherence()
    coherences.append(coherence_score)

# Print the average coherence score and its standard deviation
avg_coherence = sum(coherences) / len(coherences)
std_dev = (sum([(x - avg_coherence) ** 2 for x in coherences]) / len(coherences)) ** 0.5
print('Average coherence score:', avg_coherence)
print('Standard deviation:', std_dev)