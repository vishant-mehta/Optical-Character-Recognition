from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Multiply
from tensorflow.keras.constraints import MaxNorm


class Newm():
    def create_newm(self, vocab_size):
        #input shape : [(None, 1024, 128, 1)]
        fmodel = Sequential()

        #layer1 output : (None, 512, 64, 64)
        fmodel.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform", input_shape=(1024,128,1)))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
        
        #layer2 output : (None, 256, 32, 100)
        fmodel.add(Conv2D(filters=100, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(1,2), strides=(2,2), padding="valid"))
        
        #layer3 output : (None, 128, 16, 128)
        fmodel.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

        #layer4 output : (None, 128, 16, 128)
        fmodel.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        
        #layer5 output : (None, 128, 8, 128)
        fmodel.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid"))

        #layer6 output : (None, 128, 4, 256)
        fmodel.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid"))
        
        #layer7 output : (None, 128, 1, 256)
        fmodel.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid"))
       
        shape = fmodel.output_shape
        nb_units = shape[2] * shape[3]

        #output after reshaping : (None, 128, 256)
        fmodel.add(Reshape((shape[1], nb_units)))

        #output after BLSTM : (None, 128, 512)
        fmodel.add(Bidirectional(LSTM(units=nb_units, return_sequences=True, dropout=0.5)))

        #output after Dense : (None, 128, 512)
        fmodel.add(Dense(units=nb_units * 2))

        #output after BLSTM: (None, 128, 512)
        fmodel.add(Bidirectional(LSTM(units=nb_units, return_sequences=True)))

        #output after Dense : (None, 128, 98 )
        fmodel.add(Dense(units=vocab_size, activation="softmax"))

        return fmodel

class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""

        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config

class Flor_Model:

    def create_flor_model(self, vocab_size):
        #input shape : [(None, 1024, 128, 1)]

        # input_data = Input(name="input", shape=input_size)

        fmodel = Sequential()

        #layer1 output : (None, 512, 64, 16)
        fmodel.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform", input_shape=(1024,128,1)))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same"))

        #layer2 output : (None, 512, 64, 32)
        fmodel.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same"))

        #layer3 output : (None, 256, 16, 40)
        fmodel.add(Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2])))
        fmodel.add(Dropout(rate=0.2))

        #layer4 output : (None, 256, 16, 48)
        fmodel.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2])))
        fmodel.add(Dropout(rate=0.2))

        #layer5 output : (None, 128, 4, 56)
        fmodel.add(Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2])))
        fmodel.add(Dropout(rate=0.2))

        #layer6 output : (None, 128, 2, 64)
        fmodel.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        fmodel.add(PReLU(shared_axes=[1, 2]))
        fmodel.add(BatchNormalization(renorm=True))
        fmodel.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid"))
        shape = fmodel.output_shape
        nb_units = shape[2] * shape[3]

        #output after reshaping : (None, 128, 128)
        fmodel.add(Reshape((shape[1], nb_units)))

        #output after BGRU : (None, 128, 256)
        fmodel.add(Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5)))

        #output after Dense : (None, 128, 256)
        fmodel.add(Dense(units=nb_units * 2))

        #output after BGRU : (None, 128, 256)
        fmodel.add(Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5)))

        #output after BGRU : (None, 128, 256)
        fmodel.add(Dense(units=vocab_size, activation="softmax"))

        return fmodel


        