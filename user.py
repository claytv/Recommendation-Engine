import pandas as pd
import numpy as np
import pickle
import string
import re


import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



class User:
    '''
    User

    '''
    def __init__(self, user_id, interaction_df, content_df):

        self.user_id = user_id
        self.interaction_df = interaction_df
        self.content_df = content_df
        self.user_item = create_user_item_matrix(self.interaction_df)

    def most_popular_articles(self, n_articles, ids=False):
        if ids:
            return get_top_article_ids(n_articles, self.interaction_df)
        else:
            return get_top_articles(n_articles, self.interaction_df)

    def other_user_recommendation(self, n_recs, ids=False):
        if ids:
            return user_recs(self.user_id, self.interaction_df, self.user_item, n_recs)[0]
        else:
            return user_recs(self.user_id, self.interaction_df, self.user_item, n_recs)[1]

    def content_based_recommendation(self, n_recs, ids = False ):
        if ids:
            return make_content_recs(self.user_id, self.interaction_df, self.content_df, n_recs)[0]
        else:
            return make_content_recs(self.user_id, self.interaction_df, self.content_df, n_recs)[1]
    def similar_users(self, n_users):
        return find_similar_users(self.user_id, self.user_item)[:n_users]






def get_top_articles(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    # Your code here
    n_article_interactions = df['article_id'].value_counts().head(n)
    top_idx = n_article_interactions.index
    top_articles = list(df.loc[df['article_id'].isin(top_idx), :]['title'].unique())

    return top_articles


def get_top_article_ids(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''

    top_article_ids = list(df['article_id'].value_counts().head(n))

    return top_article_ids


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    '''
    # Fill in the function here
    user_item = df.groupby(['user_id', 'article_id'])['article_id'].count().unstack()
    user_item = user_item.fillna(0)

    for column in user_item.columns:
        user_item[column] = user_item[column].apply(lambda x: x if x == 0 else 1)
    return user_item


def find_similar_users(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first

    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered

    '''
    sim_users = np.dot(user_item, user_item.T)
    index = range(1, sim_users.shape[0] + 1)
    sim_users = pd.DataFrame(sim_users, index=index, columns=index)

    similar = sim_users[sim_users.index == user_id]

    similar = similar.drop(user_id, axis=1)
    similar = similar.T.sort_values(user_id, ascending=False)
    similar_names = similar.index

    return list(similar_names)


def user_recs(user_id, df, user_item, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    user_item - user_item_matrix to use
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found

    '''
    similar_users = get_top_sorted_users(user_id, df, user_item)
    user_articles_seen = get_user_articles(user_id, df, user_item)[0]
    recs = []
    for user in similar_users['neighbor_id']:
        articles_seen = get_user_articles(user, df, user_item)[0]
        if len(recs) < m:
            for item in articles_seen:
                if item not in user_articles_seen:
                    recs.append(item)
        else:
            break
    recs = recs[:m]
    rec_names = get_article_names(recs,df)

    return recs, rec_names


def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook
    user_item - (pandas dataframe) matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u

    Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                    highest of each is higher in the dataframe

    '''

    df_article_views = df[['user_id', 'article_id']].groupby(['user_id']).count()

    similarity = []
    for user in range(1, user_item.shape[0] + 1):
        sim = np.dot(user_item.loc[1], user_item.loc[user])
        similarity.append((user, sim))

    # sort by similarity
    similarity.sort(key=lambda x: x[1], reverse=True)

    # create dataframe
    df_sims = pd.DataFrame()
    df_sims['user_id'] = [x[0] for x in similarity]
    df_sims['similarity'] = [x[1] for x in similarity]
    df_sims = df_sims.set_index('user_id')

    # dataframe with users sorted by closest followed by most articles viewed
    neighbors_df = pd.merge(df_sims, df_article_views, on='user_id')
    neighbors_df = neighbors_df[['similarity', 'article_id']]
    neighbors_df = neighbors_df.reset_index()
    neighbors_df.columns = ['neighbor_id', 'similarity', 'num_articles']
    self_idx = neighbors_df[neighbors_df['neighbor_id'] == user_id].index
    neighbors_df = neighbors_df.drop(self_idx)
    neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_articles'], ascending=False)

    return neighbors_df


def get_user_articles(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the doc_full_name column in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # Your code here
    article_ids = user_item.columns.values[list(user_item.loc[user_id,] == 1)]
    article_ids = article_ids.astype(str)
    article_names = get_article_names(article_ids,df)
    return article_ids, article_names


def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the title column)
    '''
    # Your code here
    article_names = df.loc[df['article_id'].isin(article_ids)]['title'].unique()

    return article_names


def make_content_recs(user_id,df, df_content , n_recs=10):
    '''
    INPUT:

    OUTPUT:

    '''
    already_seen = get_user_articles(user_id)[1]
    token_titles = []

    for title in already_seen:
        clean_token = clean_and_tokenize(title)
        token_titles.append(clean_token)

    unique_token_titles = []
    for title in token_titles:
        for token in title:
            if token not in unique_token_titles:
                unique_token_titles.append(token)

    titles_df = df_content[['article_id', 'doc_full_name']]
    titles_df.index = df_content['article_id']
    titles_df = titles_df.rename(columns={"doc_full_name": "title"})
    titles_df['title_tokens'] = titles_df['title'].apply(lambda x: clean_and_tokenize(x))
    titles_df = titles_df.drop('title', axis=1)
    lam_intersect = lambda x: len(set(x).intersection(unique_token_titles))
    titles_df['user_similar_tokens'] = titles_df['title_tokens'].apply(lam_intersect)
    titles_df = titles_df.sort_values('user_similar_tokens', ascending=False)

    recs = titles_df['article_id'][:n_recs]
    recs = recs.values
    rec_names = get_article_names(recs, df=df)
    return list(recs), list(rec_names)


def clean_and_tokenize(text):
    '''
    INPUT:

    OUTPUT:

    '''
    text = re.sub(r'[^\w\s]', '', text)
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in word_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in clean_tokens if not w in stop_words]

    return filtered_tokens