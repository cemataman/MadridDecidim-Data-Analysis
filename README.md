# Decide Madrid Data Analysis

The Decide Madrid dataset consists of textual data collected from a digital participation platform in Madrid, Spain. 
This platform functions as a forum for residents to participate in ongoing urban design and planning projects. 
The dataset, which is open-source and predominantly in Spanish, is available on a platform that is accessible to the public.

## Data Description

The dataset includes textual data collected from various participation modules, wherein Madrid residents engage in debates and submit proposals. 
The data collection process, spanning multiple years, includes diverse elements such as comment and reply counts, support numbers, and tagging systems. 
The topics discussed on the platform range from design issues to policy matters. 
The participation modules are structured hierarchically, enabling users to respond to both debate descriptions and comments from other participants. 
This hierarchical structure ensures that each debate topic encompasses additional discussions related to the main topic description.

## Citation
Please include the following citation if you are interested in using the introduced methods:

Ataman, Cem, Bige Tuncer, and Simon Perrault. 2023. Transforming Large-Scale Participation Data through Topic Modelling in Urban Design Processes. In *INTERCONNECTIONS: Co-computing beyond boundaries - Proceedings of the 20th Conference on Computer-Aided Architectural Design (CAAD) Futures*, Delft, the Netherlands.

## How to Use the Code

It is advisable to adhere to the sequence of the following five steps when executing the code:

1. Execute the **data_extraction.py** file to filter and hierarchically structure the data
2. Subsequently, run **data_cleaning.py** to preprocess and clean the data.
3. Execute **data_translation_debates.py** to translate debate texts from Spanish to English.
4. Execute **data_translation_comments.py** to translate comment texts from Spanish to English.
5. Run **sentiment_analysis.py** to perform sentiment analysis on the preprocessed translated data.

Completing these five steps will yield structured data frames and results that can be utilized in subsequent analyses. 
After the data frames have been created, the topic modeling scripts may be employed according to the requirements of the analysis. 
Please make sure to replace the file path with that of your own dataset.

### Folders and Scripts

- **Scripts**: A folder containing all the relevant scripts.
    - *data_cleaning.py*: Responsible for preprocessing textual data.
    - *data_extraction.py*: Serves to filter and structure the data frame.
    - *data_translation_debates.py*: Translates debates from Spanish to English
    - *data_translation_comments.py*: Translates comments from Spanish to English
    - *sentiment_analysis.py*: Computes the sentiment scores for each argument.
    - **HDP**
        - *HDP_topic_model_and_hyperparameter_tuning.py*: trains a HDP model for topic modeling, extracts and prints topics with their key words, tests on a sample document, and generates word clouds for each topic, which are saved as images. 
        - *HDP_10_fold_cross_validation.py*: trains an HDP model for topic modeling, performs 10-fold cross-validation, calculates coherence scores for each fold, and computes the average coherence score and standard deviation to evaluate the model's performance and consistency.
    - **LDA**
        - *LDA_hyperparameter_tuning.py*: performs hyperparameter tuning on a text corpus, calculating coherence and perplexity for different combinations of hyperparameters, and logs the results for analysis.
        - *LDA_tuning_results_brief.py*: reads LDA tuning results from the CSV file, identifies the maximum coherence value for each topic count, and saves these results, along with associated hyperparameters, to a new CSV file.
        - *LDA_topic_model.py*: reads text data and a corpus from files, trains an LDA model for topic modeling with specified parameters and topic number, and prints the top words for each topic.
        - *LDA_10_fold_cross_validation.py*: uses 10-fold cross-validation to tune an LDA model, calculating coherence scores for each fold and deriving average and standard deviation across all folds.
        - *LDA_dominant_sentence.py*: trains an LDA model on text data, extracts dominant topics for each document, and saves this information along with the most representative documents for each topic into CSV files.
    - **LSI**
        - *LSI_hyperparameter_tuning.py*: performs hyperparameter tuning on a text corpus, calculating coherence for different combinations of hyperparameters, and logs the results for analysis.
        - *LSI_topic_model.py*: reads text data and a corpus from files, trains an LSI model for topic modeling with specified parameters and topic number, and prints the results for each topic.
        - *LSI_10_fold_cross_validation.py*: uses 10-fold cross-validation to tune an LSI model, calculating coherence scores for each fold and deriving average and standard deviation across all folds.
        - *LSI_dominant_sentence.py*: trains an LSI model on text data, extracts dominant topics for each document, and saves this information along with the most representative documents for each topic into CSV files.
    - *data_visualization_clustering.py*: generates and saves bar and density plots visualizing average topic contributions and document distributions from topic modeling results, using data from a CSV file.