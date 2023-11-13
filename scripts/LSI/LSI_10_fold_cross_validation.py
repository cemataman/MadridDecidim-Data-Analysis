from gensim.models import LsiModel
import numpy as np
from sklearn.model_selection import KFold
from gensim import corpora
from gensim.models import CoherenceModel
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


# Define the number of folds for cross-validation
num_folds = 10

# Define the number of topics to test for LSI model
num_topics = 13

# Define the k-fold cross-validation object
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Create a list to store the coherence scores for each fold
coherence_scores = []

# Loop through each fold
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(corpus)):
    # Split the current fold into a training set and a validation set
    train_corpus = [corpus[i] for i in train_idx]

    # Convert the validation set from tuples to lists of tokens
    val_corpus = [[id2word[word_id] for word_id, freq in doc] for doc in train_corpus]

    # Fit the LSI model on the training set
    lsimodel = LsiModel(corpus=train_corpus, num_topics=num_topics, id2word=id2word)

    # Calculate the coherence score for the LSI model on the validation set
    coherence_model_lsi = CoherenceModel(model=lsimodel, texts=val_corpus, dictionary=id2word, coherence='u_mass')
    coherence_score = coherence_model_lsi.get_coherence()
    coherence_scores.append(coherence_score)

    print(f"Fold {fold_idx + 1}: Coherence Score = {coherence_score}")

# Calculate the average coherence score and standard deviation across all folds
avg_coherence_score = sum(coherence_scores) / num_folds
std_coherence_score = np.std(coherence_scores)

print(f"Average Coherence Score = {avg_coherence_score}")
print(f"Standard Deviation = {std_coherence_score}")
