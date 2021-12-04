# from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Multiply
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import Progbar
from tensorflow.keras import backend as K
import numpy as np

class New_Model:

    def create_newm(self, vocab_size):
        #input shape : [(None, 1024, 128, 1)]
        fmodel = Sequential()

        #layer1 output : (None, 512, 64, 16)
        fmodel.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform", input_shape=(1024,128,1)))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
        
        #layer2 output : (None, 256, 32, 32)
        fmodel.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid"))
        
        #layer3 output : (None, 128, 16, 40)
        fmodel.add(Conv2D(filters=40, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

        #layer4 output : (None, 128, 16, 48)
        fmodel.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        
        #layer5 output : (None, 128, 8, 56)
        fmodel.add(Conv2D(filters=56, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid"))

        #layer6 output : (None, 128, 2, 64)
        fmodel.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid"))
        
        #layer7 output : (None, 128, 1, 128)
        fmodel.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid"))
        
        shape = fmodel.output_shape
        nb_units = shape[2] * shape[3]

        #output after reshaping : (None, 128, 128)
        fmodel.add(Reshape((shape[1], nb_units)))

        #output after BLSTM : (None, 128, 256)
        fmodel.add(Bidirectional(LSTM(units=nb_units, return_sequences=True, dropout=0.5)))

        #output after Dense : (None, 128,256)
        fmodel.add(Dense(units=nb_units * 2))

        #output after BLSTM: (None, 128, 256)
        fmodel.add(Bidirectional(LSTM(units=nb_units, return_sequences=True)))

        #output after Dense : (None, 128, 98 )
        fmodel.add(Dense(units=vocab_size, activation="softmax"))

        return fmodel


    def predict_model(self, model, x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10,
                        workers=1, use_multiprocessing=False, ctc_decode=True):

        print("Model Predict")

        out = model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps,
                                    callbacks=callbacks, max_queue_size=max_queue_size,
                                    workers=workers, use_multiprocessing=use_multiprocessing)

        # if not ctc_decode:
        #     return np.log(out.clip(min=1e-8)), []

        steps_done = 0
        print("CTC Decode")
        progbar = Progbar(target=steps)

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(x_test, x_test_len, greedy=False, beam_width=10,
                                        top_paths=1)

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            progbar.update(steps_done)

        return (predicts, probabilities)
