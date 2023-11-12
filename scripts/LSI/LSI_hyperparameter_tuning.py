from gensim.models import LsiModel, CoherenceModel
from gensim import corpora
import pickle
import csv

# Load the corpus and dictionary
x = open("/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Create a dictionary from the final data
id2word = corpora.Dictionary(final_data)

# Load the preprocessed corpus from a file
with open('/data/intermediate_data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

# Define a range of possible values for the number of topics to test
num_topics_list = range(10, 21)

# Create an empty list to store the results
results = []

# Loop over each number of topics and calculate the coherence score for an LSI model with that number of topics
for num_topics in num_topics_list:
    # Train the LSI model
    lsimodel = LsiModel(corpus=corpus, num_topics=num_topics, id2word=id2word)

    # Calculate the coherence score for the model
    coherence_model = CoherenceModel(model=lsimodel, texts=final_data, dictionary=id2word, coherence='u_mass')
    coherence_score = coherence_model.get_coherence()

    # Print the coherence score for the current number of topics
    print(f"Number of topics: {num_topics} - Coherence score: {coherence_score}")

    # Append the results to the list
    results.append([num_topics, coherence_score])

# Save the results to a CSV file
with open('/data/intermediate_data/lsi_tuning_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['num_topics', 'coherence_score'])
    for result in results:
        writer.writerow(result)