# Model

### Subclassing Implementation 

I implementation 'MnistClassifier' Model by keras subclassing with 'keras.models' and 'keras.layers'



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



---



### Manually Implementation

I tried implementation forward propagation of Model manually 

And left comment for explain about the code.

#### MnistClassifier.py

```python
from tensorflow.keras.models import Model

from Fully_connected import Dense, Flatten
from Feature_Extractor import Conv2D, MaxPooling2D

class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        # feature extractor
        # input shape : (None, 28, 28, 1)
        self.Conv2D1 = Conv2D(filters = 8, kernel_size = 5, padding = 'same')
        # shape : (None, 28, 28, 8)
        self.MaxPool1 = MaxPooling2D(pool_size = 3, strides = 2)
        # shape : (None, 14, 14, 8)
        self.Conv2D2 = Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu')
        # shape : (None, 14, 14, 32)
        self.MaxPool2 = MaxPooling2D(pool_size = 2, strides = 2)
        # shape : (None, 7, 7, 32)

        self.Flatten = Flatten()
        # shape : (None, 1568)       
       

    def call(self, x):
        x = self.Conv2D1(x)
        x = self.MaxPool1(x)
        x = self.Conv2D2(x)
        x = self.MaxPool2(x)

        x = self.Flatten(x)
        
        # input shape : (None, 1568) 
        self.Dense_relu = Dense(x.shape[1], units = 64)
        x = self.Dense_relu.relu(x)
        # shape : (None, 64) 
        self.Dense_softmax = Dense(x.shape[1], units = 10)
        x = self.Dense_softmax.softmax(x)
        # shape : (None, 10) 

        return x
```



#### Feature_Extractor.py

In Feature_Extractor, Implemented Conv2D, MaxPooling2D layer class.

I checked the Conv2D have to receive input data by dimensions.

data.shape = (batch_size, height, width, number_of_chennal)

##### class Conv2D

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
        
        '''
        filters : number of channel
        padding : 'valid' or 'same' 
        '''

        self.weight = tf.random.normal(mean = 0, stddev = 1, shape = (kernel_size, kernel_size))
        self.bias = tf.random.normal(mean = 0, stddev = 1, shape = (1, ))  # gary scale

        self.valid_idx = None
        # window slicing 시작 자리
        self.conved_images = list()

        self.relu = Activation('relu')

    def convolution(self, x_image):
        H = x_image.shape[1]
        W = x_image.shape[2]
        #
        
        H_conved = None
        W_conved = None
        if self.padding == 'same':
            self.valid_idx = int(0)  
            H_conved = H
            W_conved = W
        elif self.padding == 'valid':	### 1.
            self.valid_idx = int((self.kernel_size-1)/2)  
            H_conved = int((H - self.kernel_size)/self.strides) + 1
            W_conved = int((W - self.kernel_size)/self.strides) + 1

        batch_size = x_image.shape[0]

        for filter_idx in range(self.filters):  
    
            for batch_idx in range(batch_size):
                conved = np.zeros(shape = (H_conved, W_conved))
                cov_r_idx = 0

                for r_idx in range(self.valid_idx, H - self.valid_idx, self.strides): 	### 2.
                    cov_c_idx = 0
                    for c_idx in range(self.valid_idx, W - self.valid_idx, self.strides): 
                        receptive_field = x_image[batch_idx, r_idx - self.valid_idx : r_idx + self.valid_idx + 1, 
                                                    c_idx - self.valid_idx : c_idx + self.valid_idx + 1]
                        
                        receptive_field = receptive_field.squeeze()
                        conved_tmp = receptive_field*self.weight  
                        # the weight means element of kernel
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

      
```



**### 1.**

```python
elif self.padding == 'valid':
            self.valid_idx = int((self.kernel_size-1)/2)  
            H_conved = int((H - self.kernel_size)/self.strides) + 1
            W_conved = int((W - self.kernel_size)/self.strides) + 1
```

 change height of Feature map in convolution and pooling layer conform to this formula
$$
H_n = \left[ \frac{H_{n-1} + 2p - k}{s}  \right] + 1 \\
W_n = \left[ \frac{W_{n-1} + 2p - k}{s}  \right] + 1
$$
`$H_{n-1}$` : height of image before input Conv2D(or Pooling)

`$H_{n}$` : height of image of output from Conv2D(or Pooling)

width of image computes same way



**### 2.**

```python
for r_idx in range(self.valid_idx, H - self.valid_idx, self.strides): 	
                    cov_c_idx = 0
                    for c_idx in range(self.valid_idx, W - self.valid_idx, self.strides): 
                        receptive_field = x_image[batch_idx, r_idx - self.valid_idx : r_idx + self.valid_idx + 1, 
                                                    c_idx - self.valid_idx : c_idx + self.valid_idx + 1]
                        
                        receptive_field = receptive_field.squeeze()
                        conved_tmp = receptive_field*self.weight  
                        # the weight means element of kernel
                        conved_tmp = np.sum(conved_tmp) + self.bias

                        conved[cov_r_idx,cov_c_idx] = conved_tmp
```



**Convolution process**

![](https://blog.kakaocdn.net/dn/sSowL/btqCLeODqbH/0VNYdYkafga04UZhgxliv0/img.png)

Each element of output is add bias to computed sum after element wise multiplication between Receptive Field of Image and Filter. Than, move right direction as many steps as stride. And iterate this moving until the filter to reach last index of image.

Calculation process of pooling is same 



##### class MaxPooling2D

```python
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

##### class Dense

```python
import tensorflow as tf

from AffineFunction import Layer
from tensorflow.feras.layers import Activation

class Dense:
    def __init__(self, feature_dim, units):
        self._feature_dim = feature_dim

        self._affine= Layer(self._feature_dim, units) ### 1.
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

```



**### 1.**

```python
self._affine= Layer(self._feature_dim, units)
```

Detail explanation about 'Layer' write next part 



##### class Flatten

```python
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

    def mul_node(self, x, y): ### 1.
        self._x, self._y = x, y
        self._z = self._x*self._y
        return self._z 
    
    def plus_node(self, x, y):  ### 2.
        self._x, self._y = x, y
        self._z = self._x + self._y
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

I implementation the layer using `plus node`, `muntiplication node`.



**### 1.**

```python
    def mul_node(self, x, y): ### 1.
        self._x, self._y = x, y
        self._z = self._x*self._y
        return self._z 
```

In here, `x` means input data. And `y` means weight.



**### 2.**

```python
    def plus_node(self, x, y):  ### 2.
        self._x, self._y = x, y
        self._z = self._x + self._y
        return self._z
```

In here, `x` means (data×weight). And `y` means bias.

we can implementation forward propagation of deep learning using `for loop`

```python
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
```



---



**plus comment - backward propagation Implementation**

we can implementation backward propagation using 'chain rule'

![](https://github.com/HibernationNo1/TIL/blob/master/image/25.jpg?raw=true)



**basic_node.py**

```python
import numpy as np

class plus_node():
    def forward(self, x, y):
        self._x, self._y = x, y
        self._z = self._x + self._y
        return self._z
    
    def backward(self, dz):
        return (dz, dz)

class mul_node():
    def forward(self, x, y):
        self._x, self._y = x, y
        self._z = self._x*self._y
        return self._z

    def backward(self, dz):
        return (dz*self._y, dz*self._x)

class minus_node():
    def forward(self, x, y):
        self._x, self._y = x, y
        self._z = self._x - self._y
        return self._z

    def backward(self, dz):
        return (-1*dz, -1*dz)

class square_node():
    def forward(self, x):
        self._x = x
        self._z = self._x*self._x
        return self._z

    def backward(self, dz):
        return (2*dz*self._x)

class mean_node():
    def forward(self, x):
        self._x = x
        self._z = np.mean(self._x)
        return self._z

    def backward(self, dz):
        dx = dz*1/len(self._x)*np.ones_like(self._x)
        return dx 
```



**Affine_Function.py**

```python
import numpy as np
import basic_node as nodes

class Affine_Function:
    def __init__(self, feature_dim, Th):
        self._feature_dim = feature_dim
        self._Th = Th
        self._dth_sum_list = self._Th.copy()

        self._z1_list = [None]*(self._feature_dim + 1)
        self._z2_list = self._z1_list.copy()
        self._dz1_list, self._dz2_list = self._z1_list.copy(), self._z1_list.copy()
        self._dth_list = self._z1_list.copy()
        
        self.affine_imp()

    def affine_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]

    def forward(self, x):
        for node_idx in range(1, self._feature_dim + 1):
            self._z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], x[:, node_idx])
        
        for node_idx in range(1, self._feature_dim + 1):
            if node_idx == 1:
                self._z2_list[node_idx] = self._node2[node_idx].forward(self._Th[0], self._z1_list[node_idx])
            else :
                self._z2_list[node_idx] = self._node2[node_idx].forward(self._z2_list[node_idx-1], self._z1_list[node_idx])

        return self._z2_list[-1]

    def backward(self, dz2_last, lr):
        for node_idx in reversed(range(1, self._feature_dim + 1)): 
            if node_idx == 1:
                self._dth_list[0], self._dz1_list[node_idx] = self._node2[node_idx].backward(self._dz2_list[node_idx])
            elif node_idx == self._feature_dim:
                self._dz2_list[node_idx -1], self._dz1_list[node_idx] = self._node2[node_idx].backward(dz2_last)
            else: 
                self._dz2_list[node_idx -1], self._dz1_list[node_idx] = self._node2[node_idx].backward(self._dz2_list[node_idx])
        
            self._dth_sum_list[0] = np.sum(self._dth_list[0])
        for node_idx in reversed(range(1, self._feature_dim + 1)): 
            self._dth_list[node_idx], _ = self._node1[node_idx].backward(self._dz1_list[node_idx])
            self._dth_sum_list[node_idx] = np.sum(self._dth_list[node_idx])

        self._Th = self._Th - (lr*np.array(self._dth_sum_list).reshape(-1, 1))
        return self._Th
```

