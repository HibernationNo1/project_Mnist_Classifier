from tensorflow.keras.models import Model

from Fully_connected import Dense, Flatten
from Feature_Extractor import Conv2D, MaxPooling2D

class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        # feature extractor
        self.Conv2D1 = Conv2D(filters = 8, kernel_size = 5, padding = 'same')
        # (None, 28, 28, 8)
        self.MaxPool1 = MaxPooling2D(pool_size = 2, strides = 2)
        # (None, 14, 14, 8)
        self.Conv2D2 = Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu')
        # (None, 14, 14, 32)
        self.MaxPool2 = MaxPooling2D(pool_size = 2, strides = 2)
        # (None, 7, 7, 32)

        self.Flatten = Flatten()
        # (None, 1568)       
       

    def call(self, x):
        x = self.Conv2D1(x)
        x = self.MaxPool1(x)
        x = self.Conv2D2(x)
        x = self.MaxPool2(x)

        x = self.Flatten(x)
        
        self.Dense_relu = Dense(x.shape[1], units = 64)
        x = self.Dense_relu.relu(x)
        self.Dense_softmax = Dense(x.shape[1], units = 10)
        x = self.Dense_softmax.softmax(x)

        return x
