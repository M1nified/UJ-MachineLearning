# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Lab10/hw'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
#  ## Homework 10
#
#  ### Your task now is to construct a neural network (in Keras) that learns clasification on this data set. Try to optimize and regularize it with the methods you have learned in class and by going thrugh this notebook. What is the best validation dataset score you can achieve?
#
#  Here's what you should take away from this example:
#
#  - If you are trying to classify data points between N classes, your network should end with a Dense layer of size N.
#  - In a single-label, multi-class classification problem, your network should end with a softmax activation, so that it will output a probability distribution over the N output classes.
#  - Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the network, and the true distribution of the targets.
#  - There are two ways to handle labels in multi-class classification: Encoding the labels via "categorical encoding" (also known as "one-hot encoding") and using categorical_crossentropy as your loss function. Encoding the labels as integers and using the sparse_categorical_crossentropy loss function.
#  - If you need to classify data into a large number of categories, then you should avoid creating information bottlenecks in your network by having intermediate layers that are too small.

# %%
# Imports

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, optimizers, regularizers
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical


# %%

train, test = reuters.load_data(num_words=10000)
train_data, train_labels = train
test_data, test_labels = test


# %%
print(len(train_data), len(train_labels))


# %%
print(len(test_data), len(test_labels))


# %%
train_data[1]


# %%
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])
# Note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# %%
decoded_newswire


# %%
train_labels


# %%
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
x_train.shape, x_test.shape


# %%
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# Our vectorized training labels
one_hot_train_labels = to_one_hot(train_labels)
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)


one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# %% [markdown]
#  ## The homework part

# %%
# Prep


def model_compile(model):
    global optimizer
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


def model_fit(model):
    global x_train, x_test, one_hot_train_labels, num_epochs
    model_history = model.fit(
        x_train,
        one_hot_train_labels,
        validation_data=(x_test, one_hot_test_labels),
        epochs=num_epochs,
        batch_size=512,
        verbose=1
    )
    return model_history


def model_evaluate(model):
    global x_test, one_hot_test_labels
    model_score = model.evaluate(
        x_test,
        one_hot_test_labels,
        verbose=1
    )
    return model_score


def print_score(model_score, idx=None, title=None):
    if idx is not None:
        print("Model", idx)
    if title is not None:
        print(title)
    print("Accuracy: %.2f%%" % (model_score[1]*100))
    print("Test loss:", model_score[0])
    print("Test accuracy", model_score[1])


# %%
last_layer_size = len(list(set(train_labels)))
last_layer_size


# %%
input_size = len(x_train[0])
first_layer_size = 500
input_size, first_layer_size

# %%
model_1 = models.Sequential()
model_1.add(layers.Dense(
    first_layer_size,
    activation='relu',
    input_shape=(input_size,)
))
model_1.add(layers.Dense(
    last_layer_size,
    activation='softmax'
))


optimizer = optimizers.RMSprop(
    lr=0.001,
    rho=0.9,
    epsilon=None,
    decay=0.0
)


# %%
model_compile(model_1)
num_epochs = 10
model_1_history = model_fit(model_1)
model_1_score = model_evaluate(model_1)


# %%
print_score(model_1_score, idx=1)

# %%
list_epochs = range(1, num_epochs+1)
plt.figure(figsize=(16, 10))
plt.plot(
    list_epochs,
    model_1_history.history['loss'],
    'b',
    label='Training loss'
)
plt.plot(
    list_epochs,
    model_1_history.history['val_loss'],
    'r',
    label='Validation loss'
)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# %%
plt.figure(figsize=(16, 10))
plt.plot(
    list_epochs,
    model_1_history.history['acc'],
    'b',
    label='Training accuracy'
)
plt.plot(
    list_epochs,
    model_1_history.history['val_acc'],
    'r',
    label='Validation accuracy'
)
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %% [markdown]
#  ## Model with smaller layers

# %%
first_layer_size = 100
model_2 = models.Sequential()
model_2 = models.Sequential()
model_2.add(layers.Dense(
    first_layer_size,
    activation='relu',
    input_shape=(input_size,)
))
model_2.add(layers.Dense(
    last_layer_size,
    activation='softmax'
))


# %%
model_compile(model_2)
model_2_history = model_fit(model_2)
model_2_score = model_evaluate(model_2)


# %%
print_score(model_1_score, idx=1)


# %%
print_score(model_2_score, idx=2)

# %% [markdown]
#  ### Differences
#  For smaller layers the accuracy rised just a little, but the loss rate got much better.
#  Model 1 reaches low training loss, but overfits sooner.

# %%
list_epochs = range(1, num_epochs+1)
plt.figure(figsize=(16, 10))
plt.plot(
    list_epochs,
    model_1_history.history['loss'],
    'b--',
    label='Training loss1'
)
plt.plot(
    list_epochs,
    model_1_history.history['val_loss'],
    'm--',
    label='Validation loss1'
)
plt.plot(
    list_epochs,
    model_2_history.history['loss'],
    'r-',
    label='Training loss2'
)
plt.plot(
    list_epochs,
    model_2_history.history['val_loss'],
    'c-',
    label='Validation loss2'
)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
#  ## Model with dropout

# %%
model_3 = models.Sequential()
model_3.add(layers.Dense(
    first_layer_size,
    activation='relu',
    input_shape=(input_size,)
))
model_3.add(layers.Dropout(0.5))
model_3.add(layers.Dense(
    last_layer_size,
    activation='softmax'
))


# %%
model_compile(model_3)
model_3_history = model_fit(model_3)
model_3_score = model_evaluate(model_3)

# %%
print_score(model_2_score, idx=2)

# %%
print_score(model_3_score, idx=3)

# %% [markdown]
#  ### Differences
#  Model with dropout has:
#  - better validation loss
#  - bigger training loss
#

# %%
list_epochs = range(1, num_epochs+1)
plt.figure(figsize=(16, 10))
plt.plot(
    list_epochs,
    model_2_history.history['loss'],
    'b--',
    label='Training loss no drop'
)
plt.plot(
    list_epochs,
    model_2_history.history['val_loss'],
    'm--',
    label='Validation loss no drop'
)
plt.plot(
    list_epochs,
    model_3_history.history['loss'],
    'r-',
    label='Training loss with drop'
)
plt.plot(
    list_epochs,
    model_3_history.history['val_loss'],
    'c-',
    label='Validation loss with drop'
)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
#  ## Model with L2 regularization

# %%
model_4 = models.Sequential()
model_4.add(layers.Dense(
    first_layer_size,
    kernel_regularizer=regularizers.l2(0.01),
    activation='relu',
    input_shape=(input_size,)
))
model_4.add(layers.Dropout(0.5))
model_4.add(layers.Dense(
    last_layer_size,
    activation='softmax'
))


# %%
model_compile(model_4)
model_4_history = model_fit(model_4)
model_4_score = model_evaluate(model_4)


# %%
print_score(model_3_score, idx=3)


# %%
print_score(model_4_score, idx=5)

# %% [markdown]
#  ### Differences
#  L1 regularization worsened the model.
#

# %% [markdown]
#  ## Model with L1 regularization

# %%
model_5 = models.Sequential()
model_5.add(layers.Dense(
    first_layer_size,
    kernel_regularizer=regularizers.l1(0.01),
    activation='relu',
    input_shape=(input_size,)
))
model_5.add(layers.Dropout(0.5))
model_5.add(layers.Dense(
    last_layer_size,
    activation='softmax'
))


# %%
model_compile(model_5)
model_5_history = model_fit(model_5)
model_5_score = model_evaluate(model_5)


# %%
print_score(model_3_score, idx=3)


# %%
print_score(model_5_score, idx=5)

# %% [markdown]
#  ### Differences
#  Model with L1 regularization got much more worse than with only L2.
#
# %% [markdown]
#  ## Model with L1 and L2 regularizations

# %%
model_6 = models.Sequential()
model_6.add(layers.Dense(
    first_layer_size,
    kernel_regularizer=regularizers.l1(0.01),
    activation='relu',
    input_shape=(input_size,)
))
model_6.add(layers.Dropout(0.5))
model_6.add(layers.Dense(
    first_layer_size,
    kernel_regularizer=regularizers.l2(0.01),
    activation='relu',
    input_shape=(input_size,)
))
model_6.add(layers.Dropout(0.5))
model_6.add(layers.Dense(
    last_layer_size,
    activation='softmax'
))


# %%
model_compile(model_6)
model_6_history = model_fit(model_6)
model_6_score = model_evaluate(model_6)


# %%
print_score(model_3_score, idx=3)


# %%
print_score(model_6_score, idx=6)

# %% [markdown]
# ### Differences
# Applying both L1 and L2 regularizations also worsened the model results.
#

# %% [markdown]
# ## Conclusion
# The best results were achieved with Model 3, which is the one with small layers and dropout applied.
#
