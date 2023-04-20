import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import recommender



def plot_similarity():
    movie=input("Enter movie:")
    list,scores=recommender.recommender(movie)
    if list=="Not available.Try some other movie":
        return list
    fig = plt.figure(figsize = (20, 5))
    # creating the bar plot
    print(list,scores)
    plt.bar(list, scores, color ='maroon',width = 0.4)
    plt.xlabel("Movie Title")
    plt.ylabel("Similarity Score")
    plt.title("Top ten recommended movies for the movie: {}".format(movie.title()))
    plt.xticks(rotation=45)
    plt.show()
plot_similarity()
