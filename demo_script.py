import user
import pandas as pd
from user import User


df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']
df_content = df_content.drop_duplicates(subset='article_id', keep = 'first')


def email_mapper():
    '''
    OUTPUT:
    a column with the corresponding user_id to each email 
    Description:
    This is a data wrangling function that assigns every unique email a unique
        user id which makes the data easier to work with later
    '''
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

user10 = User(10, df, df_content)

print("\nThe top 10 trending articles are: ")
for article in user10.most_popular_articles(10, ids=False):
    print(article)

print("\nThe 10 article ids for the user based recommendation are: ")
for article_id in user10.other_user_recommendation(10,ids=True):
    print(article_id)

print("\nThe 5 article ids for the content based recommendation are: ")
for article in user10.content_based_recommendation(5):
    print(article)

print("\nThe 7 most similar user ids are: " )
for user_id in user10.similar_users(7):
    print(user_id)




