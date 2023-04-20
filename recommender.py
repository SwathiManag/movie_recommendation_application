import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data=pd.read_csv('main_data.csv')
vectorizer=CountVectorizer()
count_matrix=vectorizer.fit_transform(data['comb'])
similarity_matrix=cosine_similarity(count_matrix)


def recommender(movie):
    movie=movie.lower()
    if movie not in data['movie_title'].unique():
        return("Not available.Try some other movie"),0
    index=data.loc[data['movie_title']==movie].index[0]
    movie_list=list(enumerate(similarity_matrix[index]))
    movie_list=sorted(movie_list, key = lambda x:x[1] ,reverse=True)
    movie_list=movie_list[1:11]
    similar_movies=[]
    for i in range(len(movie_list)):
        a=movie_list[i][0]
        similar_movies.append(data['movie_title'][a])
    scores = []
    for i in range(len(movie_list)):
        scores.append(movie_list[i][1])
    
    return similar_movies,scores

def string_to_list(str):
    li = str.split('","')
    li[0] = li[0].replace('["','')
    li[-1] = li[-1].replace('"]','')
    return li