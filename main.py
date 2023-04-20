import numpy as np
import pandas as pd
import pickle
from recommender import recommender,string_to_list
from review_scraping import reviews
from flask import Flask, render_template, request


data=pd.read_csv('main_data.csv')
model_pickel = 'naive_bayes.pkl'
model = pickle.load(open(model_pickel, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = list(data['movie_title'].str.capitalize())
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc,score= recommender(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = list(data['movie_title'].str.capitalize())

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = string_to_list(rec_movies)
    rec_posters = string_to_list(rec_posters)
    cast_names = string_to_list(cast_names)
    cast_chars = string_to_list(cast_chars)
    cast_profiles = string_to_list(cast_profiles)\
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i],cast_profiles[i]] for i in range(len(cast_profiles))}
    
    movie_reviews=reviews(model,vectorizer,imdb_id)
    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts)

if __name__ == '__main__':
    app.run(debug=True)
