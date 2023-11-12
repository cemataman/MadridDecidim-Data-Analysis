import pandas as pd
import gensim.utils
from gensim import corpora
import pickle

# Load preprocessed data from a file
with open("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/final_data.py", "r") as x:
    final_data = eval(x.readlines()[0])

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)


from gensim.models import LsiModel
lsi_model = LsiModel(corpus=corpus, id2word=id2word, num_topics=13)


def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, doc_topics in enumerate(ldamodel[corpus]):
        doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        for j, (topic_num, prop_topic) in enumerate(doc_topics):
            if j == 0:  # dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                new_row = pd.DataFrame([[int(topic_num), round(prop_topic, 4), topic_keywords]], columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
                sent_topics_df = pd.concat([sent_topics_df, new_row], ignore_index=True)
            else:
                break
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


# Get the dominant topic for each document
df_topic_sents_keywords = format_topics_sentences(ldamodel=lsi_model, corpus=corpus, texts=final_data)

# Reset the index and set column names
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Group the DataFrame by dominant topic and get the document with the highest percentage contribution for each topic
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], axis=0)

# Reset the index and set column names for the sorted DataFrame
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Save the DataFrames to CSV files
df_dominant_topic.to_csv("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_dominant_topic.csv", index=False)
sent_topics_sorteddf_mallet.to_csv("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/results/LSI_sorted_topics.csv", index=False)


