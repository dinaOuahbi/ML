# %% [markdown]
# ## Naive Bayes Algorithm

# %% [markdown]
# **Theory** 
# 
# Bayes theorem is one of the earliest probabilistic inference algorithms
# 
# we are computing the probability of an event(a person being a threat) based on the probabilities of certain related events(age, sex, presence of bag or not, nervousness etc. of the person).
# 
# One thing to consider is the independence of these features amongst each other.
# 
# This is the 'Naive' bit of the theorem where it considers each feature to be independent of each other which may not always be the case and hence that can affect the final judgement.
# 
# In short, the Bayes theorem calculates the probability of a certain event happening(in our case, a message being spam) based on the joint probabilistic distributions of certain other events(in our case, a message being classified as spam).
# 
# **DATA**
# 
# The first column takes two values, 'ham' which signifies that the message is not spam, and 'spam' which signifies that the message is spam.
# 
# The second column is the text content of the SMS message that is being classified
# 
# 

# %%
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
df = pd.read_table('data/SMSSpamCollection',names=['label','sms_message'])
df.head()

# %%
# convert labels to numeric
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

# %%
#split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

# convert our data into the desired matrix format

"""
The basic idea of Bag of Words is to take a piece of text and count the frequency of the words in that text
"""
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# we will be using the multinomial Naive Bayes algorithm : suitable for classification with discrete features
#On the other hand, Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data has a Gaussian (normal) distribution.

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
predictions.shape


# %%
# evaluate how well our model is doing
    #Accuracy :  True Positives / total predictions
    #Precision : True Positives/(True Positives + False Positives)
    # Recall : True Positives/(True Positives + False Negatives)
    #F1 score : weighted average of the precision and recall scores


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))


