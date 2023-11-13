import random
import numpy as np
from sklearn.model_selection import KFold
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pickle

# Load the corpus and dictionary
x = open("/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)


### HYPERPARAMETER TUNING -------------------------------------------------------------------------------
# Set the number of folds
num_folds = 10

# Shuffle the corpus
random.shuffle(corpus)

# Split the corpus into num_folds equal parts
folds = list(KFold(n_splits=num_folds, shuffle=False).split(corpus))

# Initialize a list to store the coherence scores for each fold
coherence_scores = []

# Loop through each fold
for fold_idx, (train_indexes, val_indexes) in enumerate(folds):
    # Split the current fold into a training set and a validation set
    train_corpus = [corpus[i] for i in train_indexes]

    # Convert the validation set from tuples to lists of tokens
    val_corpus = [[id2word[word_id] for word_id, freq in doc] for doc in train_corpus]

    # Train an LDA model on the training set
    lda_model = LdaModel(corpus=train_corpus,
                         id2word=id2word,
                         num_topics=15,
                         random_state=42,
                         update_every=8,
                         chunksize=100,
                         passes=10,
                         alpha=0.31,
                         eta=0.91,
                         per_word_topics=True)

    # Calculate the coherence score for the validation set
    coherence_model_lda = CoherenceModel(model=lda_model, texts=val_corpus, dictionary=id2word, coherence='u_mass')
    coherence_score = coherence_model_lda.get_coherence()
    coherence_scores.append(coherence_score)

    print(f'Fold {fold_idx + 1} coherence score: {coherence_score}')

# Calculate the average coherence score and standard deviation across all folds
avg_coherence_score = np.mean(coherence_scores)
std_coherence_score = np.std(coherence_scores)

# Print out the results
print(f'Average coherence score across {num_folds} folds: {avg_coherence_score}')
print(f'Standard deviation of coherence scores across {num_folds} folds: {std_coherence_score}')
