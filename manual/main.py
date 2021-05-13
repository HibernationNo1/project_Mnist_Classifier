import os
from numpy.lib.twodim_base import tri
import tensorflow as tf

from model_class.MnistClassifier import MnistClassifier

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

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
n_class = 10 

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

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate = learning_rate)

metric_objects = get_classification_metrics()

model = MnistClassifier()

for epoch in range(start_epoch, epochs):
    
    go_train(train_ds, model, loss_object, optimizer, metric_objects)
    if epoch == epochs-1:
        con_mat = go_validation(validation_ds, model, loss_object, metric_objects, con_mat)
    else:
        _ = go_validation(validation_ds, model, loss_object, metric_objects, con_mat)
    
    go_test(test_ds, model, loss_object, metric_objects, path_dict)

    training_reporter(epoch, losses_accs, 
                      metric_objects, dir_name)
    save_losses_model(epoch, model, losses_accs, path_dict)
    
    loss_acc_visualizer(epoch, losses_accs, path_dict)

    resetter(metric_objects)

confusion_matrix_visualizer(con_mat, n_class, path_dict)
       

