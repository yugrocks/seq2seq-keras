from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model


# some parameters
output_len = 20
input_len = 20
n_a = 500 # encoder hidden size
n_s = 500 # decoder hidden size
embed_len = 50

print("Preparing Model (Without Attention).")
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Error in softmax.')


decoder_lstm_1 = LSTM(n_s, return_state=True,return_sequences=True,name = "decoder_lstm")
decoder_lstm_2 = LSTM(n_s, return_sequences=True,return_state=True, name="decoder_lstm_2")
decoder_lstm_3 = LSTM(n_s, return_sequences=True,return_state=True, name="decoder_lstm_3")
decoder_lstm_4 = LSTM(n_s, return_state=True, name="decoder_lstm_4")
output_layer = Dense(embed_len, activation="linear", name="decoder_output")

def model(input_len, output_len, n_a, n_s, embed_len):
    X = Input(shape=(input_len, embed_len),name="encoder_input")
    decoder_input = Input(shape=(input_len, embed_len), name="decoder_input") # teacher forcing with first token as SOS
    outputs = []
    
    # now the encoder LSTM
    a, _, c = LSTM(n_a, return_sequences=True,return_state=True ,name="lstm_1_encoder")(X) # a contains all the hidden states of each timestep
    s_2, _2, c_2 = LSTM(n_a, return_sequences=True,name="lstm_2_encoder", return_state=True)(a)
    s_3, _3, c_3 = LSTM(n_a,return_sequences=True,name="lstm_3_encoder", return_state=True)(s_2)
    s_4, _4, c_4 = LSTM(n_a,name="lstm_4_encoder", return_state=True)(s_3)
    # now iterate for each timestep in the decoder
    for t in range(output_len):
        # t'th input words
        t_th = Lambda(lambda x: tf.slice(x, [0,t,0], [-1,1,-1]))(decoder_input)
        s, _, c = decoder_lstm_1(t_th, initial_state=[_,c])
        s_2, _2, c_2 = decoder_lstm_2(s, initial_state=[_2, c_2])
        s_3, _3, c_3 = decoder_lstm_3(s_2, initial_state=[_3,c_3])
        s_4, _4, c_4 = decoder_lstm_4(s_3, initial_state=[_4,c_4])
        out = output_layer(s_4)
        outputs.append(out)
    model = Model(inputs = [X,decoder_input], outputs=outputs)
    return model

model = model(input_len, output_len, n_a, n_s, embed_len)
     
model.summary()
#plot_model(model, to_file='seq2seq.png') # plot the model in an image
adam = Adam(lr=0.0001)
model.compile(loss="cosine_proximity", optimizer=adam,metrics=["cosine", "accuracy"])
print("Model Loaded. Ready to begin training.")
