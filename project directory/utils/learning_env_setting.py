import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

# CONTINUE_LEARNING = True      training이 진행되다 의도치 않게 멈춰서 
#                               지난 Model의 training을 이어서 진행할 때
# CONTINUE_LEARNING = False     training을 처음부터 다시 시작할 때 (또는 처음 시작할 때)
# dir_name = project name 아니면 하위 directory name 

# ---------project에 관한 file이나 directory를 생성하는 함수---------
def dir_setting(dir_name, CONTINUE_LEARNING):
    cp_path = os.path.join(os.getcwd() , dir_name)
    # 현재 경로에서 새 directory 경로 생성
    # os.getcwd() 의 return이 상위 directory라서 path 추가함
    model_path = os.path.join(cp_path, 'model')
    # cp_path 경로 안에 새롭게 생성할 directory
    
    if CONTINUE_LEARNING == False and os.path.isdir(cp_path):
        # training을 처음부터 다시 시작하는데 기존의 정보가 남아있을 때
        shutil.rmtree(cp_path)
        # cp_path 경로의 모든 file delete

    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
    
    path_dict = {'cp_path' :cp_path, 
                'model': model_path}
    return path_dict



# ---------train, validation, test 각각의 loss와 accuracy를 담은 dict생성 함수---------
def get_classification_metrics():
    train_loss = Mean()                     # training에 사용할 loss function
    train_acc = SparseCategoricalAccuracy() # accuracy에 사용할 function
    # function_name1 = function1()

    validation_loss = Mean()                # validation에 사용할 loss function
    validation_acc = SparseCategoricalAccuracy()

    test_loss = Mean()                # test에 사용할 loss function
    test_acc = SparseCategoricalAccuracy()


    metric_objects = dict()
    metric_objects['train_loss'] = train_loss
    metric_objects['train_acc'] = train_acc
    metric_objects['validation_loss'] = validation_loss
    metric_objects['validation_acc'] = validation_acc
    metric_objects['test_loss'] = test_loss
    metric_objects['test_acc'] = test_acc

    return metric_objects

# 이렇게 여러 함수를 여기에 넣어놓으면
# Function_bank['function_name1'] 
# 이런 식으로 함수를 자유롭게 꺼낼 수 있다. 

# ---------model learning이 의도치 않게 멈췄던 학습을 마지막 stop순간부터 다시 learning 하는 함수---------
def continue_setting(CONTINUE_LEARNING, path_dict, model):
    if CONTINUE_LEARNING == True and len(os.listdir(path_dict['model_path'])) == 0:
        # CONTINUE_LEARNING가 True지만, model directory에 아무것도 없다면
        CONTINUE_LEARNING = False
        print("CONTINUE_LEARNING flag has been converted to FALSE") 

    if CONTINUE_LEARNING == True:
        # ----- model return -----
        epoch_list = os.listdir(path_dict['model_path']) 
        # 'model_path' 라는 object에 저장된 경로에 남아있는 과거 training data를 전부 가져온다.
        #  그리고 training data의 각 directory 이름이 epoch_n 이라 할 때
        epoch_list = [int(epoch.split('_')[1]) for epoch in epoch_list]
        # epoch.split('_') = [epoch, n] 이기 때문에 n = 1 이라고 하면 위 코드는 
        # epoch_list = [1 for epoch in epoch_list] 이 되는 것이다.
        epoch_list.sort()

        last_epoch = epoch_list[-1]
        model_path = path_dict['model_path'] + '/epoch_' + str(last_epoch)
        # 마지막 epoch의 경로를 model_path에 저장
        model = tf.keras.models.load_model(model_path)
        # 경로에 있는 model을 return한다.


        losses_accs_path = path_dict['train1']
        losses_accs_np = np.load(losses_accs_path +'/losses_accs.npz')
        # project construct 파일의 구조를 보면 어떤 경로의 data를 가져왔는지 이해 될거임
        losses_accs = dict()
        for k, v in losses_accs_np.items():
            losses_accs[k] = list(v)
            # losses_acc_np의 값을 새로운 dict에 저장

        start_epoch = last_epoch + 1 
        # 다시 시작할 training 시작점 설정
    else:
        model = model
        start_spoch = 0
        losses_acc = {'train_losses': [], 'train_accs': [],
                        'validation_losses': [], 'validation_accs': []}
        # 빈 dict를 만들어준다.

    return model, losses_accs, start_epoch