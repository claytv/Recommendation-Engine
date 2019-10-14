# Recommendation-Engine
Recommendation engine created for the Udacity Data Scientist Nanodegree


# Overview 
The goal of this project was to get an introduction into the different kinds of recommendations and how they work. The project was sponsored by IBM and they provided the data, one file being records of every user-article interaction and the other being information about each article. The project through Udacity required us to implement rank based and user-user based recommendation functions. In addition to these functions I chose to implement a content based recommendation and combined all of these functions into the 'User' class.  

# Files
* Data - Folder containg data in .csv files
* demo_script.py - Example using the User class
* exploratory_notebook.ipynb - Notebook where I developed the functions originally 
* install_nltk.py - File to solve the issue I had with using the nltk.download() function 
* user.py - File containing the class 'User' 

# Data
## Interaction Data

## Content Data

# Recommendations
## Rank Based Recommendation 
This recommendation returns the most popular articles throughout all of the users and is used when recommending articles to users who have not yet read any articles. 

## User-User Recommendation 
This recommendation looks at users with similar reading habits and finds articles that the similar users have seen but the user recieving the recommendation has not. 

## Content Based Recommendation 
This recommendation tokenizes every article that a user has watched and then compares unique tokens with other articles the user has not seen. The articles with the most similarties are recommended. 

* I could have included the words in the article but I felt that simplifying and using only the title was appropriate for this situation
* This recommendation has a bias towards articles with longer title names
* This recommendation does not properly account for users who view multiple articles with similarly worded titles

https://stackoverflow.com/questions/41929044/ssl-certificate-verify-failed-certificate-verify-failed-ssl-c-661
