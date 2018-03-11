path_output = r"/output/"
from seq2seq import *
from  generator import MasterGenerator
from preprocess_data import *
import pickle
import keras


class NBatchLogger(keras.callbacks.Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print(self.display,"Batches have completed")
            print(self.params["metrics"])
            
    def on_epoch_end(self, epoch, logs={}):
        print(epoch,"Epochs end")
        print(logs)

train_gen = MasterGenerator(file_X_train,file_y_train,word_vectors, batch_size=1).get_generator()
test_gen = MasterGenerator(file_X_test,file_y_test,word_vectors, batch_size=1).get_generator()

print("LOADING WEIGHTS")
model.load_weights("weights/weights1.h5")
print("WEIGHTS LOADED")

print("Starting Training.")
out_batch = NBatchLogger(display=1000)
model.fit_generator(train_gen,samples_per_epoch=len(file_X_train)//32,validation_data=test_gen,
                         validation_steps=len(file_X_test)//32
                        ,epochs=25,initial_epoch=20,verbose=0,callbacks=[out_batch])


model.save_weights(path_output+"weights1.h5") # save weights as a whole
model.save("whole_model.hdf5") # saving the whole model with weights
