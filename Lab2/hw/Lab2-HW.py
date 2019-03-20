# %% [markdown]
# ## Dataset info
#
# http://help.sentiment140.com/for-students/
#
# ```
# The data is a CSV with emoticons removed. Data file format has 6 fields:
# 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# 1 - the id of the tweet (2087)
# 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
# 4 - the user that tweeted (robotickilldozr)
# 5 - the text of the tweet (Lyx is cool)
# ```

# %%
from scipy.sparse import csc_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import operator
import functools
import numpy as np
from nltk.tokenize import word_tokenize
import csv

# %%

train_data = []
with open('training.1600000.processed.noemoticon.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        newrow = [int(row[0]), row[5]]
        train_data.append(newrow)

train_positive_list = list(
    map(lambda r: r[1], filter(lambda r: r[0] == 4, train_data)))
train_positive_list = train_positive_list[:1000]
train_positive = '\n'.join(train_positive_list)
train_nonpositive_list = list(
    map(lambda r: r[1], filter(lambda r: not r[0] == 4, train_data)))
train_nonpositive_list = train_nonpositive_list[:1000]
train_nonpositive = '\n'.join(train_nonpositive_list)

# print(train_data) # won't work in jupyter due to amount of data

test_data = []
with open('testdata.manual.2009.06.14.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        newrow = [int(row[0]), row[5]]
        test_data.append(newrow)

test_positive_list = list(
    map(lambda r: r[1], filter(lambda r: r[0] == 4, test_data)))
test_positive = '\n'.join(test_positive_list)
test_nonpositive_list = list(
    map(lambda r: r[1], filter(lambda r: not r[0] == 4, test_data)))
test_nonpositive = '\n'.join(test_nonpositive_list)

real_values = [1 if tweet[0] == 4 else 0 for tweet in test_data]  # for testing

# %%
features = word_tokenize(test_positive + " " + test_nonpositive)
features = list(set(features))

# %%


class NaiveBayes:

    def __init__(self, training_sets, labels):
        self.labels = labels
        self.k = len(self.labels)
        words = [word_tokenize(text) for text in training_sets]
        word_count = sum(len(word) for word in words)
        self.likelihoods = [{word: (words[i].count(word) + 0.5) /
                             len(words[i]) + 0.5 for word in words[i]} for i in range(self.k)]
        self.priors = [len(words[i]) / word_count for i in range(self.k)]

    def predict(self, input_text):
        probs = self._probabilities(input_text)
        best_index = np.argmax(probs)
        return self.labels[best_index]

    def _probabilities(self, text):
        words = word_tokenize(text)
        probs = [self._probability_for_class(words, i) for i in range(self.k)]
        return probs

    def _probability_for_class(self, words, k):
        probs_for_words = map(
            lambda word: self.likelihoods[k][word] if word in self.likelihoods[k] else np.finfo(np.float32).eps, words)
        prob_for_text = functools.reduce(operator.mul, probs_for_words, 1)
        return prob_for_text


# %%
nb = NaiveBayes([train_positive, train_nonpositive], [1, 0])


# %%

predictions = [nb.predict(tweet[1]) for tweet in test_data]

my_conf_mat = confusion_matrix(real_values, predictions)
print(my_conf_mat)

plt.figure()
sns.heatmap(my_conf_mat)
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.figure()
my_conf_mat_normalized = my_conf_mat.astype(
    'float') / my_conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(my_conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')

# %%
my_accuracy_score = accuracy_score(real_values, predictions)
my_accuracy_score

# %%
my_precision_score = precision_score(real_values, predictions)
my_precision_score


# %%
my_recall_score = recall_score(real_values, predictions)
my_recall_score

# %% [markdown]
# ## Same with sklearn

# %%
x_train_row_zeros = [0] * len(features)
x_train = [x_train_row_zeros[:]
           for _ in range(len(train_positive_list) + len(train_nonpositive_list))]

train_all_list = train_positive_list[:]
train_all_list.extend(train_nonpositive_list[:])

for x_i, tweet in enumerate(train_all_list):
    twords = word_tokenize(tweet)
    utwords = list(set(twords))
    for word in utwords:
        try:
            count = twords.count(word)
            idx = features.index(word)
            x_train[x_i][idx] = count
        except:
            pass

y_train = [True] * len(train_positive_list)
y_train.extend([False] * len(train_nonpositive_list))

# %%

x_test_row_zeros = [0] * len(features)
x_test = []
x_test = [x_test_row_zeros[:]
          for _ in range(len(test_positive_list) + len(test_nonpositive_list))]

test_all_list = test_positive_list[:]
test_all_list.extend(test_nonpositive_list[:])

for x_i, tweet in enumerate(test_all_list):
    twords = word_tokenize(tweet)
    utwords = list(set(twords))
    for word in utwords:
        try:
            count = twords.count(word)
            idx = features.index(word)
            x_test[x_i][idx] = count
        except:
            pass

y_test = []
y_test = [True] * len(test_positive_list)
y_test.extend([False] * len(test_nonpositive_list))

# %%
clf = MultinomialNB()
clf.fit(x_train, y_train)


# %%
predictions_from_sklearn = clf.predict(x_test)
# predictions_from_sklearn


# %%
sklearn_conf_mat = confusion_matrix(y_test, predictions_from_sklearn)
print(sklearn_conf_mat)

# %%
sklearn_accuracy_score = accuracy_score(y_test, predictions_from_sklearn)
sklearn_accuracy_score

# %%
sklearn_precision_score = precision_score(y_test, predictions_from_sklearn)
sklearn_precision_score


# %%
sklearn_recall_score = recall_score(y_test, predictions_from_sklearn)
sklearn_recall_score


# %% [markdown]
# ## Comparison

# %%
print(my_conf_mat)
print(sklearn_conf_mat)

# %%
print(my_accuracy_score, sklearn_accuracy_score)
print(my_precision_score, sklearn_precision_score)
print(my_recall_score, sklearn_recall_score)

# %% [markdown]
# ## Summary

# Custom model seems to work only a bit worse than the one from sklearn library. However, it improves as we add more training data (feature coverage) and becomes better than sklearn's one.


# %%
