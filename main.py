#there are 3 types of recommendation system
#Content based recommendation system -> recommends based on the content you watch
#Popularity based recommendation system -> recommends based on popularity
#Collaborative recommendation system -> groups people based on what they watch. for example there is a group of people who watch particular type of movie and then a new user starts watching those types of movies as well then it will start recommending the new user the type of movies the group of people watches.

#workflow
#data -> data-preprocessing -> feature Extraction(to convert textual data into numerical values) -> user input -> cosine similarity algorithm (we will convert each movies into vectors and find the similarity between them using cosine similarity algorithm where it finds the angle between two non null vectors) -> will get a list of movies and then provide them to the user

#importing dependencies
import numpy as np
import pandas as pd
import difflib #used for comparing sequences, specifically strings and generating differences between them. For example the user inputs a name of a movie with a spelling mistake the computer needs to compare and find the best match for the inputted string
from sklearn.feature_extraction.text import TfidfVectorizer #used for converting textual data into numerical values
from sklearn.metrics.pairwise import cosine_similarity #algorithm which we are going to use

#data collection and pre-processing
#loading the data from csv file to a pandas dataframe
movies_data = pd.read_csv('movies.csv')
#print(movies_data.head())
#print(movies_data.tail())
#print(movies_data.shape)

#selecting the relevant features for recommendations
selected_features = ['genres','keywords','tagline','cast','director']
#print(selected_features)

#replacing the null values with null strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('') #fillna represents fill which are not available

#combining all the 5 selected features
#means we will combine all the genres, keywords,tagline etc together for that movie
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

#print(combined_features)

#converting the text data into feature vectors
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features) #this code will fit and transform all the features into numerical data

#print(feature_vectors)

#getting the similarity score or similarity confidence value using cosine similarity
similarity = cosine_similarity(feature_vectors) #this will take the numerical values and see which movies are similar to each other based on the numerical values

#print(similarity)

#print(similarity.shape)

#will ask the user for the input and check if its present in the dataset

movie_name = input("Enter the name of the movie : ")

#should now give back the movies similar to the one given by the user
#creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist() #tolist() will take all the values and create a list

# print(list_of_all_titles)

#finding the close match for the movie name given by the user by using difflib and comparing it with all the names in the list

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles) 
# print(find_close_match)

#we want only one match and not all
close_match = find_close_match[0]
#print(close_match)

#finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0] #it will be in the form of a list. Thats why we used [0]

#print(index_of_the_movie)

#getting the list of similar movies based on the index values

similarity_score = list(enumerate(similarity[index_of_the_movie])) #enumerate is used to run a loop in a list.useful for returning an indedexed list. this command here will take the index of the movie given by the user and compare it with aother movies and return a list of the similarity_score. list because there are several movies so it will return different similarity_score corresponding to the index
#print(similarity_score)

#sorting the movies based on similarity_score(from highest to lowest)
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) #reverse = True means we want to sort it from highest to lowest. lambda function is used as the key for sorting. It defines how the sorting should be performed. In this case, it's sorting based on the second element of each pair in similarity_score (the similarity score). The lambda function takes an element x(which is the similarity_score) from the list and returns x[1], which is the similarity score. x[0] is the index from the list

#print(sorted_similar_movies)

#taking 10-20 movies from the list and print the name of it corresponding to the index

print('Movies suggested for you based on genres, cast and directors: \n')
#creating a for loop
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if (i<=20):
        print(i,'.',title_from_index)
        i+=1




