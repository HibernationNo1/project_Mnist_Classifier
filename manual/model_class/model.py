from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import dense, flatten

class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        # feature extractor


        self.fe.add(Conv2D(filters = 8, kernel_size = 5, padding = 'same', activation = 'relu'))  
        # (None, 28, 28, 8)
        self.fe.add(MaxPooling2D(pool_size = 2, strides = 2))
        # (None, 14, 14, 8)
        self.fe.add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu'))  
        # (None, 14, 14, 32)
        self.fe.add(MaxPooling2D(pool_size = 2, strides = 2))
        # (None, 7, 7, 32)

        self.Flatten = flatten()
        # (None, 1568)       
       

    def call(self, x):
        x = self.fe(x)
        

        x = self.Flatten(x)
        
        self.Dense_relu = dense(x.shape[1], units = 64)
        x = self.Dense_relu(x)
        self.softmax = dense(x.shape[1], units = 10)
        x = self.softmax(x)

        return x
