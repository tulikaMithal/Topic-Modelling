# Topic-Modelling
# Topic-Modelling

Dataset used: https://yelp.com/dataset/documentation/json

Analysis 1:

a) Extracted  the restaurants data in Phoenix city of Arizona state. 
b) Further narrowed down upon the restaurants with categories: Italian & Pizza.
c) Once we got the filtered data in step (b), we found the resaturant with the highest average rating and the one with the lowest average rating.
d) Did topic modelling for the review text of both, the best and worst restaurants.

Analysis 2:
a) Extracted the restaurant’s data (reviews and corresponding star rating).
b) Tokenized the review text into words.
c) Removed stopwords.
d) Hashed the sentence into feature vectors.
e)  Used Inverse Document Frequency (IDF)  to rescale the feature vectors
f) Trained a model using Logistic Regression classification technique, thereby predicting the star rating given a new review text.

Steps to run the code:

a) Part 1 (Topic Modelling)
Command to run:
spark-submit --class "TopicModelling" <jar name> <path to review.json> <path to business.json> <output filepath>

b) Part 2 (Topic Modelling)
Command to run:
spark-submit --class "StarPrediction" <jar name> <path to review.json> <path to business.json> <output filepath>

