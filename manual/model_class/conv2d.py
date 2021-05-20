import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

filters = 8, kernel_size = 5, padding = 'same', activation = 'relu'))

class Conv2D:
    def __init__(self, filters, kernel_size, padding, stride = 1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = tf.random.normal(mean = 0, stddev = 1, shape = (kernel_size, kernel_size))
        self.bias = tf.random.normal(mean = 0, stddev = 1, shape = (1, ))  # gary scale

        self.valid_idx = None
        # window slicing 시작 자리

    def convolution(self, x):
        H = x.shape[1]
        W = x.shape[2]
        H_conved = None
        W_conved = None
        if self.padding == 'same':
            self.valid_idx = int(0)  
            H_conved = H
            W_conved = W
        elif self.padding == 'valid':
            self.valid_idx = int((self.kernel_size-1)/2)  
            H_conved = int((H - self.kernel_size)/self.stride) + 1
            W_conved = int((W - self.kernel_size)/self.stride) + 1

        x_image = x.numpy().squeeze()
        conved = np.zeros(shape = (H_conved, W_conved))
        for r_idx in range(valid_idx, H - 1 - valid_idx):
            for c_idx in range(valid_idx, W - 1 - valid_idx): 
                receptive_field = x_image[r_idx - valid_idx : r_idx + valid_idx + 1, 
                                            c_idx - valid_idx : c_idx + valid_idx + 1]
                # print(receptive_field.shape)  
                # filter shape대로 receptive_field가 잘 설정됨을 알 수 있다.
                conved_tmp = receptive_field*self.weight  # weight
                conved_tmp = np.sum(conved_tmp) + self.bias

                conved[r_idx - valid_idx : r_idx + valid_idx + 1, 
                            c_idx - valid_idx : c_idx + valid_idx + 1] = conved_tmp
        return conved

    def pooing:


num_of_kernel = 1
pad = 0
kernel_size = 3



valid_idx = int((kernel_size-1)/2)  
# padding  = 0 이므로 를 valid_idx로 설정

test_image = test_image.numpy().squeeze()


print(f"W_conved: {W_conved} , H_conved : {W_conved}") 


# window slicing
for r_idx in range(valid_idx, H - 1 - valid_idx ):
    for c_idx in range(valid_idx, W - 1 - valid_idx): 
        receptive_field = test_image[r_idx - valid_idx : r_idx + valid_idx + 1, 
                                     c_idx - valid_idx : c_idx + valid_idx + 1]
        # print(receptive_field.shape)  
        # filter shape대로 receptive_field가 잘 설정됨을 알 수 있다.
        conved_tmp = receptive_field*w  # weight
        conved_tmp = np.sum(conved_tmp) + b

        conved[r_idx - valid_idx : r_idx + valid_idx + 1, 
                      c_idx - valid_idx : c_idx + valid_idx + 1] = conved_tmp

print(f"conved {conved}")  
