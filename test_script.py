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
    Creates
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

user1 = User(1, df, df_content)

print(user1.other_user_recommendation(10)[1])


