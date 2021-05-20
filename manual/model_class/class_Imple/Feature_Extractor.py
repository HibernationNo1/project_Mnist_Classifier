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
                        # print(receptive_field.shape)  
                        # filter shape대로 receptive_field가 잘 설정됨을 알 수 있다.
                        receptive_field = receptive_field.squeeze()
                        conved_tmp = receptive_field*self.weight  # weight
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
                        # print(receptive_field.shape)  
                        # filter shape대로 receptive_field가 잘 설정됨을 알 수 있다.
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

