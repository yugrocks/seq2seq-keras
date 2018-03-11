import gensim
import numpy as np
import nltk

path_input = r"data/"

print("loading Glove vectors...")
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        path_input+'glove.6B.50d.w2vformat_2.txt', binary=False)
print("Glove vectors loaded.")

# downloading punkt model
nltk.download('punkt')

print("Loading data.")
# some hyperparameters
sentence_length = 20
file_X_train = []
file_y_train = []
file_X_test = []
file_y_test = []

with open(path_input+"train.enc",encoding="utf8",errors='ignore') as X_train:
    file_X_train = X_train.readlines()
    
with open(path_input+"train.dec", encoding="utf8",errors='ignore') as y_train:
    file_y_train = y_train.readlines()
    
with open(path_input+"test.enc",encoding="utf8",errors='ignore') as X_test:
    file_X_test = X_test.readlines()

with open(path_input+"test.dec",encoding="utf8",errors='ignore') as y_test:
    file_y_test = y_test.readlines()


# convert each to list
for i in range(len(file_X_train)):
    file_X_train[i] = nltk.tokenize.word_tokenize(file_X_train[i].lower())
    file_y_train[i] = nltk.tokenize.word_tokenize(file_y_train[i].lower())
    
for i in range(len(file_X_test)):
    file_X_test[i] = nltk.tokenize.word_tokenize(file_X_test[i].lower())
    file_y_test[i] = nltk.tokenize.word_tokenize(file_y_test[i].lower())

for i in range(len(file_X_train)):
    if len(file_X_train[i]) > sentence_length:
        file_X_train[i] = file_X_train[i][0:sentence_length]
    if len(file_y_train[i]) > sentence_length:
        file_y_train[i] = file_y_train[i][0:sentence_length]

for i in range(len(file_X_test)):
    if len(file_X_test[i]) > sentence_length:
        file_X_test[i] = file_X_test[i][0:sentence_length]
    if len(file_y_test[i]) > sentence_length:
        file_y_test[i] = file_y_test[i][0:sentence_length]

print("Data Loaded")
