import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle


def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity


m=input("Enter movie:")
m=m.lower()
data,similarity=create_similarity()
if m not in data['movie_title'].unique():
    print('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
else:
    i = data.loc[data['movie_title']==m].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
    lst = lst[1:11] # excluding first item since it is the requested movie itself
    l = []
    for i in range(len(lst)):
        a = lst[i][0]
        l.append(data['movie_title'][a])
    print(l)
    # plot showing top ten movies based on cosine similarity score
    scores = []
    for i in range(len(lst)):
        scores.append(lst[i][1])

    fig = plt.figure(figsize = (20, 5))
 
    # creating the bar plot
    plt.bar(l, scores, color ='maroon',
            width = 0.4)
        
    plt.xlabel("Movie Title")
    plt.ylabel("Similarity Score")
    plt.title("Top ten recommended movies for the movie: {}".format(m.title()))
    plt.xticks(rotation=45)
    plt.show()
