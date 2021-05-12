import os
from numpy.lib.twodim_base import tri
import tensorflow as tf


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import  Flatten, Dense, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy


from utils.learning_env_setting import dir_setting, get_classification_metrics
from utils.dataset_utils import load_processing_mnist
from utils.cp_utils import save_losses_model, loss_acc_visualizer
from utils.cp_utils import confusion_matrix_visualizer 
from utils.basic_utils import resetter, training_reporter
from utils.train_validation_test import go_train, go_validation, go_test

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# --- Hyperparameter Setting ---
CONTINUE_LEARNING = False
dir_name = 'train1'
n_class = 10 # for confusion matrix

start_epoch = 0

train_ratio = 0.8
train_batch_size, test_batch_size = 32, 128

epochs = 30
learning_rate = 0.01

# ------------------

path_dict = dir_setting(dir_name, CONTINUE_LEARNING)
train_ds, validation_ds, test_ds = load_processing_mnist(train_ratio, 
                                                        train_batch_size, 
                                                        test_batch_size)

losses_accs = {'train_losses': [], 'train_accs': [], 
                'validation_losses': [], 'validation_accs': []}
con_mat = tf.zeros(shape = (n_class, n_class), dtype = tf.int32)
# confusion_matrix

# ---model implementation---
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

    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)
        return x

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate = learning_rate)

metric_objects = get_classification_metrics()

model = MnistClassifier()

for epoch in range(start_epoch, epochs):
    
    go_train(train_ds, model, loss_object, optimizer, metric_objects)
    go_validation(validation_ds, model, loss_object, metric_objects, con_mat)
    
    go_test(test_ds, model, loss_object, metric_objects, path_dict)

    training_reporter(epoch, losses_accs, 
                      metric_objects, dir_name)
    save_losses_model(epoch, model, losses_accs, path_dict)
    
    loss_acc_visualizer(losses_accs, path_dict)
    confusion_matrix_visualizer(con_mat, n_class, path_dict)

    resetter(metric_objects)
    
       

