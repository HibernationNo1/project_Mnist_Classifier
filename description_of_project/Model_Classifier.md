# Model_Classifier

### Subclassing Implementation 

I implementation 'MnistClassifier' Model by keras subclassing using 'keras.models' and 'keras.layers'



```python
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import  Flatten, Dense

class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        # feature extractor
        self.fe = Sequential()

        self.fe.add(Conv2D(filters = 8, kernel_size = 5, padding = 'same', activation = 'relu'))  
        # (None, 28, 28, 8)
        self.fe.add(MaxPooling2D(pool_size = 2, strides = 2))
        # (None, 14, 14, 8)
        self.fe.add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu'))  
        # (None, 14, 14, 32)
        self.fe.add(MaxPooling2D(pool_size = 2, strides = 2))
        # (None, 7, 7, 32)

        self.classifier = Sequential()
        self.classifier.add(Flatten())
        # (None, 1568)       
        self.classifier.add(Dense(units = 64, activation = 'relu'))
        # (None, 64) 
        self.classifier.add(Dense(units = 10, activation = 'softmax'))
		# (None, 10) 
        
    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)
        return x
```

input image size is 28 × 28, shape is (None, 28, 28, 1)

'None' is for batch size, the reason why there is 1 channel is that the input image is grayscale.



| Input image | (None, 28, 28, 1)  |              |
| ----------- | ------------------ | ------------ |
|             |                    | Conv2D       |
| Feature map | (None, 28, 28, 8)  |              |
|             |                    | MaxPooling2D |
| Feature map | (None, 14, 14, 8)  |              |
|             |                    | Conv2D       |
| Feature map | (None, 14, 14, 32) |              |
|             |                    | MaxPooling2D |
| Feature map | (None, 7, 7, 32)   |              |
|             |                    | Flatten()    |
| Feature map | (None, 1568)       |              |
|             |                    | Dense        |
| Feature map | (None, 64)         |              |
|             |                    | Dense        |
| Feature map | (None, 10)         |              |

 change shape of Feature map conformed to this formula
$$
H_n = \left[ \frac{H_{n-1} + 2p - k}{s}  \right] + 1 \\
W_n = \left[ \frac{W_{n-1} + 2p - k}{s}  \right] + 1
$$
`$H_{n-1}$` : height of image before input Conv2D(or Pooling)

`$H_{n}$` : height of image of output from Conv2D(or Pooling)

width of image computes same way



### Manually Implementation

I tried implementation Model manually 



#### MnistClassifier.py

```python
from tensorflow.keras.models import Model

from Fully_connected import Dense, Flatten
from Feature_Extractor import Conv2D, MaxPooling2D

class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        # feature extractor
        self.Conv2D1 = Conv2D(filters = 8, kernel_size = 5, padding = 'same')
        # (None, 28, 28, 8)
        self.MaxPool1 = MaxPooling2D(pool_size = 3, strides = 2)
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
```



#### Feature_Extractor.py

```python
import tensorflow as tf
import numpy as np

from tensorflow.feras.layers import Activation


class Conv2D:
    def __init__(self, filters, kernel_size, padding, stride = 1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = stride

        self.weight = tf.random.normal(mean = 0, stddev = 1, shape = (kernel_size, kernel_size))
        self.bias = tf.random.normal(mean = 0, stddev = 1, shape = (1, ))  # gary scale

        self.valid_idx = None
        # window slicing 시작 자리
        self.conved_images = list()

        self.relu = Activation('relu')

    def convolution(self, x_image):
        H = x_image.shape[1]
        W = x_image.shape[2]
        H_conved = None
        W_conved = None
        if self.padding == 'same':
            self.valid_idx = int(0)  
            H_conved = H
            W_conved = W
        elif self.padding == 'valid':
            self.valid_idx = int((self.kernel_size-1)/2)  
            H_conved = int((H - self.kernel_size)/self.strides) + 1
            W_conved = int((W - self.kernel_size)/self.strides) + 1

        batch_size = x_image.shape[0]

        for filter_idx in range(self.filters):
    
            for batch_idx in range(batch_size):
                conved = np.zeros(shape = (H_conved, W_conved))
                cov_r_idx = 0

                for r_idx in range(self.valid_idx, H - self.valid_idx, self.strides):
                    cov_c_idx = 0
                    for c_idx in range(self.valid_idx, W - self.valid_idx, self.strides): 
                        receptive_field = x_image[batch_idx, r_idx - self.valid_idx : r_idx + self.valid_idx + 1, 
                                                    c_idx - self.valid_idx : c_idx + self.valid_idx + 1]
                        
                        receptive_field = receptive_field.squeeze()
                        conved_tmp = receptive_field*self.weight  
                        conved_tmp = np.sum(conved_tmp) + self.bias

                        conved[cov_r_idx,cov_c_idx] = conved_tmp
                        
                        cov_c_idx +=1
                    cov_r_idx +=1
                conved = conved[np.newaxis,:, :]
                if batch_idx == 0:
                    conved_img_tmp = conved
                else:
                    conved_img_tmp = np.vstack([conved_img_tmp, conved])
            if filter_idx == filter-1:
                self.conved_images.append(conved_img_tmp)
                self.conved_images = np.stack(self.conved_images, axis=3)
            else:
                self.conved_images.append(conved_img_tmp)
        
        return self.relu(self.conved_images)



class MaxPooling2D:
    def __init__(self, pool_size, strides):
        self.pool_size = pool_size
        self.strides = strides

        self.valid_idx = None
        self.pooled_images = list()

        self.relu = Activation('relu')

    def pooing(self, x_image):
        H = x_image.shape[1]
        W = x_image.shape[2]
        self.valid_idx = int((self.pool_size-1)/2)  
        H_pooled = int((H - self.pool_size)/self.strides) + 1
        W_pooled = int((W - self.pool_size)/self.strides) + 1

        batch_size = x_image.shape[0]
        filters = x_image.shape[3]
        for filter_idx in range(self.filters):
            for batch_idx in range(batch_size):
                pooled = np.zeros(shape = (H_pooled, W_pooled))
                pol_r_idx = 0
                
                for r_idx in range(self.valid_idx, H - self.valid_idx, self.strides):
                    pol_c_idx = 0
                    for c_idx in range(self.valid_idx, W - self.valid_idx, self.strides): 
                        receptive_field = x_image[batch_idx, r_idx - self.valid_idx : r_idx + self.valid_idx + 1, 
                                                    c_idx - self.valid_idx : c_idx + self.valid_idx + 1]
                        
                        receptive_field = receptive_field.squeeze()
                        pooled_tmp = np.max(receptive_field)
                        pooled[pol_r_idx,pol_c_idx] = pooled_tmp
                        
                        pol_c_idx +=1
                    pol_r_idx +=1
                pooled = pooled[np.newaxis,:, :, np.newaxis]
                if batch_idx == 0:
                    pooled_img_tmp = pooled
                else:
                    pooled_img_tmp = np.vstack([pooled_img_tmp, pooled])
            if filter_idx == filter-1:
                self.pooled_images.append(pooled_img_tmp)
                self.pooled_images = np.stack(self.conved_images, axis=3)
            else:
                self.pooled_images.append(pooled_img_tmp)

        return self.relu(self.pooled_images)       
```





#### Fully_connected.py

```python
import tensorflow as tf

from AffineFunction import Layer
from tensorflow.feras.layers import Activation

class Dense:
    def __init__(self, feature_dim, units):
        self._feature_dim = feature_dim

        self._affine= Layer(self._feature_dim, units)
        self._relu = Activation('relu')
        self._softmax = Activation('softmax') 

    def relu(self, x):
        z = self._affine.layer(x)
        pred = self._relu(z)
        return pred

    def softmax(self, x):
        z = self._affine.layer(x)
        pred = self._softmax(z)
        return pred

class Flatten:
    def __init__(self, x):
        self._re = x.shape[1]*x.shape[2]*x.shape[3]
        self._x = x.reshape([-1, self._re])
        return x
```



#### AffineFunction.py

```python
import numpy as np

class AffineFunction:
    def __init__(self, feature_dim, units):
        self._feature_dim = feature_dim
        self.units = units

        self._z1_list = [None]*(self._feature_dim + 1)
        self._z2_list = self._z1_list.copy()

        self.node_imp()
        self.random_theta_initialization()

    def random_theta_initialization(self):
        r_feature_dim = 1/np.power(self._feature_dim, 0.5)
        self._Th = np.random.uniform(low = -1*r_feature_dim, 
                                    high = r_feature_dim, 
                                    size =(self._feature_dim + 1, 1))
        # 초기 ramdom theta의 범위 = 1/np.power(self._feature_mid, 0.5)

    def node_imp(self):
        self._node1 = [None] + [self.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [self.plus_node() for _ in range(self._feature_dim)]

    def plus_node(self, x, y):
        self._x, self._y = x, y
        self._z = self._x + self._y
        return self._z

    def mul_node(self, x, y):
        self._x, self._y = x, y
        self._z = self._x*self._y
        return self._z 

    def perseptron(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._z1_list[node_idx] = self._node1[node_idx](self._Th[node_idx], 
                                                                    X[node_idx]) 
        # 곱셈                                                              
        self._z2_list[1] = self._node2[1](self._Th[0], self._z1_list[1]) # bias
        for node_idx in range(2, self._feature_dim + 1): 
            self._z2_list[node_idx] = self._node2[node_idx](self._z2_list[node_idx-1], 
                                                                    self._z1_list[node_idx])
        # 덧셈
        return self._z2_list[-1]     


class Layer:
    def __init__(self, feature_dim, units):
        self.feature_dim = feature_dim
        self.units = units
        self.persep_layer = None
    def layer(self, X):
        for idx in range(self.units):
            self.AffineFunction = AffineFunction(self.feature_dim, self.units)
            if idx == 0:
                self.persep_layer = self.AffineFunction.perseptron(X)
            else: 
                self.persep_layer = np.hstack([self.persep_layer, self.AffineFunction.perseptron(X)])
        return self.persep_layer
```

