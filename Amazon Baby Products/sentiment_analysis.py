import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# 1. Reading file - Collecting data

data = pandas.read_csv('amazon_baby.csv')

# 2. Preprocessing

# a) converting all reviews to strings

data['review'] = data['review'].apply(str)

# 3. Small data anaylisis

# a) find out total number of reviews for every product

groupby_name = data.groupby(['name']).size()

dictOfNameAndCount = dict(zip(groupby_name.keys(), groupby_name.get_values()))

# b) find which product has max number of reviews

print(sorted(dictOfNameAndCount.items(), key=lambda x: x[1])[-1])

# c) find over all sentiment for all reviews using bar graph

groupby_ratings = data.groupby(['rating']).size()

plt.bar(groupby_ratings.keys(), groupby_ratings.get_values()/sum(groupby_ratings.get_values())*100)

plt.show()

# 4. Applying logistic regression

# a) Considering only those reviews that do not have a rating of 3, i.e., removing neutral reviews

data = data[data['rating']!=3]

data['sentiment'] = data.rating.apply(lambda x: 1 if x>=4 else 0)

# b) creating a logistic regressor

lr = LogisticRegression()

# c) Using count vectorizer to remove all stop words and create word count dictionary

c = CountVectorizer(stop_words = "english")

data_cv = c.fit_transform(data['review'])

# d) creating test and train data

x_train, x_test, y_train, y_test = train_test_split(data_cv.toarray(), data['sentiment'], train_size = 0.05, test_size = 0.01)

# e) fitting the ml model with training data

lr.fit(x_train, y_train)

# f) applying them model to test data and predicting the score of the model

print(lr.score(x_test, y_test))

