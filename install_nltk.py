# I had some trouble using the nltk.download() function. I kept recieving a certificate error. 
# I found a solution on stack overflow the link is in README.md

import nltk

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
