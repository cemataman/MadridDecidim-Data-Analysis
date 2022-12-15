import warnings
import logging
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np

### Dataframe display settings
desired_width=520
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

### Functions for reading csv/xlsx files

#filter the discussion based on comment and vote counts
def read_madrid_debates (fn = '/Users/cem_ataman/Dropbox/Research/Collaborations/Participation_Datasets/Madrid_Data/debates.xlsx'):
    min_comment = input('please write the minimum comment count (5-10) for each discussion: ')
    min_vote = input('please write the minimum vote count (10-20) for each discussion: ')
    df = pd.read_excel(fn, sheet_name='debates', index_col=False, usecols=['id','description', 'cached_votes_total', 'comments_count', 'author_name'])

    # parameters to filter unimportant debates
    df2 = df[(df['comments_count'] > int(min_comment)) & (df['cached_votes_total'] > int(min_vote))]
    return(df2.iloc[:,[0]])

def flatten(l):
    return [item for sublist in l for item in sublist]

df = read_madrid_debates()
df2 = df.values.tolist()
debate_id = flatten(df2)

# choose the module (debate or proposal)
def read_madrid_comments (fn = '/Users/cem_ataman/Dropbox/Research/Collaborations/Participation_Datasets/Madrid_Data/comments.xlsx'):
    df = pd.read_excel(fn, sheet_name='comments', index_col=False, usecols=[ 'commentable_id', 'commentable_type', 'body'])
    df2 = df[(df['commentable_type'] == 'Debate') & (df['commentable_id'].isin(debate_id))]
    return(df2)

selected_comments = read_madrid_comments()

### to save the final dataframe as an xlsx file
selected_comments.to_excel("selected_comments_madrid.xlsx", index=False)

