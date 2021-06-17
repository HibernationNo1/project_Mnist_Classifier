from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import  Flatten, Dense

class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        # feature extractor
        self.fe = Sequential()

        self.fe.add(Conv2D(filters = 8, kernel_size = 5, padding = 'same', activation = 'relu'))  
        self.fe.add(MaxPooling2D(pool_size = 2, strides = 2))
        self.fe.add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu'))  
        self.fe.add(MaxPooling2D(pool_size = 2, strides = 2))
        
        self.classifier = Sequential()
        self.classifier.add(Flatten())     
        self.classifier.add(Dense(units = 64, activation = 'relu'))
        self.classifier.add(Dense(units = 10, activation = 'softmax'))

    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)
        return x
