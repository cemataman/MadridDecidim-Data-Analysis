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
def read_madrid_debates (fn = '/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/madrid_data/debates.xlsx'):
    df = pd.read_excel(fn, sheet_name='debates',
                       index_col=False,
                       usecols=['debate_id', 'description', 'cached_votes_total', 'comments_count', 'author_name'])

    min_com = df['comments_count'].min()
    max_com = df['comments_count'].max()
    min_comment = input(f'please write the minimum comment count ( {min_com} - {max_com} ) for each discussion: ')

    min_vot = df['cached_votes_total'].min()
    max_vot = df['cached_votes_total'].max()
    min_vote = input(f'please write the minimum vote count ( {min_vot} - {max_vot} ) for each discussion: ')

    # parameters to filter unimportant debates
    df2 = df[(df['comments_count'] > int(min_comment)) & (df['cached_votes_total'] > int(min_vote))]
    return(df2.iloc[:,[0]])

def flatten(l):
    return [item for sublist in l for item in sublist]

df = read_madrid_debates() # get the selected debate ids based on min vote and comment counts

df2 = df.values.tolist() # convert the dataframe into a list of lists
debate_id = flatten(df2) # convert the dataframe into a list of values

# choose the module (debate or proposal)
def read_madrid_comments (fn = '/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/madrid_data/comments.xlsx'):
    df = pd.read_excel(fn, sheet_name='comments', index_col=False, usecols=[ 'comment_id', 'commentable_id', 'commentable_type', 'ancestry', 'body'])
    df2 = df[(df['commentable_type'] == 'Debate') & (df['commentable_id'].isin(debate_id))]
    return(df2)

selected_comments = read_madrid_comments()

### to save the final dataframe as an xlsx file
selected_comments.to_excel("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/selected_debate_comments.xlsx", index=False)

