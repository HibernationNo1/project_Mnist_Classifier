import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras.metrics import Mean, CategoricalAccuracy

def dir_setting(dir_name, CONTINUE_LEARNING):
    cp_path = os.path.join(os.getcwd() , dir_name)

    model_path = os.path.join(cp_path, 'model')
    
    if CONTINUE_LEARNING == False and os.path.isdir(cp_path):
        shutil.rmtree(cp_path)

    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
    
    path_dict = {'cp_path' :cp_path, 
                'model_path': model_path}
    return path_dict


def get_classification_metrics():
    train_loss = Mean()                     
    train_acc = CategoricalAccuracy() 

    validation_loss = Mean()                
    validation_acc = CategoricalAccuracy()

    test_loss = Mean()               
    test_acc = CategoricalAccuracy()

    metric_objects = dict()
    metric_objects['train_loss'] = train_loss
    metric_objects['train_acc'] = train_acc
    metric_objects['validation_loss'] = validation_loss
    metric_objects['validation_acc'] = validation_acc
    metric_objects['test_loss'] = test_loss
    metric_objects['test_acc'] = test_acc

    return metric_objects


def continue_setting(CONTINUE_LEARNING, path_dict, model):
    if CONTINUE_LEARNING == True and len(os.listdir(path_dict['model_path'])) == 0:
        CONTINUE_LEARNING = False
        print("CONTINUE_LEARNING flag has been converted to FALSE") 

    if CONTINUE_LEARNING == True:
        epoch_list = os.listdir(path_dict['model_path']) 
        epoch_list = [int(epoch.split('_')[1]) for epoch in epoch_list]
        epoch_list.sort()

        last_epoch = epoch_list[-1]
        model_path = path_dict['model_path'] + '/epoch_' + str(last_epoch)
        
        model = tf.keras.models.load_model(model_path)
        
        losses_accs_path = model_path
        losses_accs_np = np.load(losses_accs_path +'/losses_accs.npz')

        losses_accs = dict()
        for k, v in losses_accs_np.items():
            losses_accs[k] = list(v)

        start_epoch = last_epoch + 1 

    else:
        model = model
        start_epoch = 0
        losses_accs = {'train_losses': [], 'train_accs': [],
                        'validation_losses': [], 'validation_accs': []}

    return model, losses_accs, start_epoch