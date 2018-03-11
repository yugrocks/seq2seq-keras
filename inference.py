from inference_model import *
from generator import MasterGenerator
import gensim
import nltk

path_input = "data/"
print("loading Glove vectors...")
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        path_input+'glove.6B.50d.w2vformat_2.txt', binary=False)
print("Glove vectors loaded.")




# load the model weights
model.load_weights("weights/weights1.h5")


def get_word_from_vector(vector):
        # return most similar word given a vector
        return word_vectors.most_similar([vector], topn=1)

def generate_words(out_sequence):
        sentence = ""
        for output in out_sequence:
            word=get_word_from_vector(output[0].reshape(50,))[0][0]
            if word == "EOS":
                break
            sentence += " "+word
        return sentence

def talk_to_me(sentence):
    X_test = [nltk.word_tokenize(sentence.lower())]
    y_test=[["SOS",],]
    g = MasterGenerator(X_test,y_test,word_vectors, batch_size=1,test=True).get_generator()
    out_sequence =  model.predict_generator(g, steps=1)
    sentence = generate_words(out_sequence)
    print("Bot:  ",sentence)
    
    
while True:
    sentence = input("yugal:      ")
    talk_to_me(sentence)
    if sentence == "quit":
        break
    
    
