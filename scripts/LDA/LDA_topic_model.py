import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gensim.utils
from gensim import corpora
from pprint import pprint
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
import pickle


x = open("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=15,
                                            random_state=42,
                                            update_every=8,
                                            chunksize=100,
                                            passes=10,
                                            alpha=0.31,
                                            eta=0.91,
                                            per_word_topics=True)


pprint (lda_model.print_topics())
#
# vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
# pyLDAvis.save_html(vis, '/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LDA_intertopic_distance_map.html')