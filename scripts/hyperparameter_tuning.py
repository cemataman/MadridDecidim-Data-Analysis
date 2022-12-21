import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gensim.utils
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import numpy as np
import tqdm

### Create a function with hyperparametes
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_data, dictionary=id2word, coherence='c_v')
    return coherence_model_lda.get_coherence()

def main():
    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 2
    max_topics = 11
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
    print(int(num_of_docs * 0.75))
    corpus_sets = [
        # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
        # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
        # gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
        corpus]
    corpus_title = ['100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }

    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=30)
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                      k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        pbar.update(1)

                        pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
        pbar.close()

if __name__ == '__main__':
    main()