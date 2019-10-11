import pandas as pd
import numpy as np

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



class User:
    '''
    User w

    '''
    def __init__(self, user_id, interaction_df, content_df):

        self.user_id = user_id
        self.interaction_df = interaction_df
        self.content_df = content_df
        self.user_item = create_user_item_matrix(self.interaction_df)

    def most_popular_articles(self, n_articles, ids=False):
        '''
        INPUT:
        n_articles - (int) number of articles to show
        ids - (boolean) True returns article ids, False returns names

        OUTPUT:
        article_ids or article_names - list of the most popular articles id or name
        '''
        if ids:
            article_ids = get_top_article_ids(n_articles, self.interaction_df)
            return article_ids
        else:
            article_names = get_top_articles(n_articles, self.interaction_df)
            return article_names

    def other_user_recommendation(self, n_recs, ids=False):
        '''
        INPUT:
        n_rec - (int) number of articles to recommend
        ids - (boolean) True returns article ids, False returns names

        OUTPUT:
        article_ids or article_names - list of the recommended articles id or name
        '''
        if len(get_user_articles(self.interaction_df,self.user_item)[0]) < 1:
            return self.most_popular_articles(n_recs, ids)
        else:
            article_ids = user_recs(self.user_id, self.interaction_df, self.user_item, n_recs)[0]
            article_names = user_recs(self.user_id, self.interaction_df, self.user_item, n_recs)[1]
            if ids:
                return article_ids
            else:
                return article_names

    def content_based_recommendation(self, n_recs, ids=False):
        '''
        INPUT:
        n_rec - (int) number of articles to recommend
        ids - (boolean) True returns article ids, False returns names

        OUTPUT:
        article_ids or article_names - list of the recommended articles id or name
        '''
        if len(get_user_articles(self.interaction_df,self.user_item)[0]) < 1:
            return self.most_popular_articles(n_recs, ids)
        else:
            article_ids = make_content_recs(self.user_id, self.interaction_df, self.content_df, self.user_item, n_recs)[0]
            article_names = make_content_recs(self.user_id, self.interaction_df, self.content_df, self.user_item, n_recs)[1]
            if ids:
                return article_ids
            else:
                return article_names

    def similar_users(self, n_users):
        '''
        INPUT:
        n_users - (int) number of similar users to return

        OUTPUT:
        similar_users - (list) list of the user ids of other users with similar reading habits
        '''
        similar_users = get_top_sorted_users(self.user_id, self.interaction_df, self.user_item)['neighbor_id'].values
        similar_users = similar_users[:n_users]
        return similar_users


# Helper functions for User class

def get_top_articles(n, df_interaction):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df_interaction - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    # Your code here
    n_article_interactions = df_interaction['article_id'].value_counts().head(n)
    top_idx = n_article_interactions.index
    top_articles = list(df_interaction.loc[df_interaction['article_id'].isin(top_idx), :]['title'].unique())

    return top_articles


def get_top_article_ids(n, df_interaction):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df_interaction - (dataframe) dataframe of user interactions with articles

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''

    top_article_ids = list(df_interaction['article_id'].value_counts().head(n))

    return top_article_ids


def create_user_item_matrix(df_interaction):
    '''
    INPUT:
    df_interaction - (dataframe) dataframe of user interactions with articles

    OUTPUT:
    user_item - (dataframe) matrix that indicates what articles each user has seen

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    '''
    # Fill in the function here
    user_item = df_interaction.groupby(['user_id', 'article_id'])['article_id'].count().unstack()
    user_item = user_item.fillna(0)

    for column in user_item.columns:
        user_item[column] = user_item[column].apply(lambda x: x if x == 0 else 1)
    return user_item


def find_similar_users(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (dataframe) matrix that indicates what articles each user has seen

    OUTPUT:
    similar_users - (list) users

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


def user_recs(user_id, df_interaction, user_item, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    df_interaction - (dataframe) dataframe of user interactions with articles
    m - (int) the number of recommendations you want for the user
    user_item - (dataframe) matrix that indicates what articles each user has seen

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Returns recommendations for a user by returning the items that users similar to them have seen
    but the user has not
    '''
    similar_users = get_top_sorted_users(user_id, df_interaction, user_item)
    user_articles_seen = get_user_articles(user_id, df_interaction, user_item)[0]
    recs = []
    for user in similar_users['neighbor_id']:
        articles_seen = get_user_articles(user, df_interaction, user_item)[0]
        if len(recs) < m:
            for item in articles_seen:
                if item not in user_articles_seen:
                    recs.append(item)
        else:
            break
    recs = recs[:m]
    rec_names = get_article_names(recs,df_interaction)

    return recs, rec_names


def get_top_sorted_users(user_id, df_interaction, user_item):
    '''
    INPUT:
    user_id - (int) user id to find similar users for
    df_interaction - (dataframe) dataframe of user interactions with articles
    user_item - (dataframe) matrix that indicates what articles each user has seen

    OUTPUT:
    neighbors_df - (dataframe)
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user
                    *sorted by similarity and num_interactions
    '''

    df_article_views = df_interaction[['user_id', 'article_id']].groupby(['user_id']).count()

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


def get_user_articles(user_id, df_interaction, user_item):
    '''
    INPUT:
    user_id - (int) user id to display articles read before
    df_interaction - (dataframe) dataframe of user interactions with articles
    user_item - (dataframe) matrix that indicates what articles each user has seen

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids
    '''
    # Your code here
    article_ids = user_item.columns.values[list(user_item.loc[user_id,] == 1)]
    article_ids = article_ids.astype(str)
    article_names = get_article_names(article_ids,df_interaction)
    return article_ids, article_names


def get_article_names(article_ids, df_interaction):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df_interaction - (dataframe) dataframe of user interactions with articles

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids
    '''
    # Your code here

    #article_names = df_interaction.loc[df_interaction['article_id'].isin(article_ids)]['title'].unique()
    article_names = df_interaction[df_interaction['article_id'].isin(article_ids)]['title'].drop_duplicates().values.tolist()

    return article_names


def make_content_recs(user_id,df_interaction, df_content, user_item, n_recs):
    '''
    INPUT:
    user_id - (int) user_id to make recommendations for
    df_interaction - (df) dataframe of user interactions with articles
    df_content - (df) dataframe of article ids and titles
    user_item - (dataframe) matrix that indicates what articles each user has seen

    OUTPUT:
    list(recs) - (list) recommended article id's
    list(rec_names) - (list) recommended article names

    Description:
    Compares words in the titles of the articles a user has seen with the titles of
    the ones he has not seen and returns the titles with the most similarity
    '''
    already_seen = get_user_articles(user_id, df_interaction, user_item)[1]
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

    already_seen_ids = get_user_articles(user_id, df_interaction, user_item)[0]
    recs = titles_df['article_id'][~titles_df['article_id'].isin(already_seen_ids)]
    recs = list(recs.values)
    rec_names = get_article_names(recs, df_interaction)

    return recs[:n_recs], rec_names[:n_recs]


def clean_and_tokenize(text):
    '''
    INPUT:
    text - (string) title to be cleaned and tokenized

    OUTPUT:
    filtered_tokens - (list) cleaned tokens
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