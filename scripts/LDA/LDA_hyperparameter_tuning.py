import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gensim
import gensim.utils
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import numpy as np
import tqdm
import pickle

# Read the final data from a file
x = open("/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)



### hyperparameter tuning ---------------------------------------------------------------
# Define a function to compute the coherence value for a given set of hyperparameters
def compute_coherence_and_perplexity(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_data, dictionary=id2word, coherence='u_mass')
    coherence_score = coherence_model_lda.get_coherence()
    perplexity_score = lda_model.log_perplexity(corpus)
    return coherence_score, perplexity_score


# Define the main function for hyperparameter tuning
def main():
    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 10
    max_topics = 20
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    # Validation sets
    num_of_docs = len(corpus)
    print(num_of_docs)

    corpus_sets = [corpus]
    corpus_title = ['100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': [],
                     'Perplexity': []
                     }

    num_iterations = len(corpus_sets) * len(topics_range) * len(alpha) * len(beta)
    print(num_iterations)

    # Iterate through different validation sets, topics, alpha and beta values
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=num_iterations)
        for i in range(len(corpus_sets)):
            for k in topics_range: # iterate through number of topics
                for a in alpha: # iterate through alpha values
                    for b in beta: # iterate through beta values

                        # get the coherence score for the given parameters
                        coherence_score, perplexity_score = compute_coherence_and_perplexity(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b)

                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(coherence_score)
                        model_results['Perplexity'].append(perplexity_score)

                        pbar.update(1) # to update the progress bar by incrementing it by 1
                        pd.DataFrame(model_results).to_csv('/data/intermediate_data/lda_tuning_results.csv', index=False)
        pbar.close()

if __name__ == '__main__':
    main()