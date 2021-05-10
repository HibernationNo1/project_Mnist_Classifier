import os
import numpy as np
from utils.learning_env_setting import dir_setting, continue_setting

dir_name = 'train1'
CONTINUE_LEARNING = True
model = CNN # 임의로 CNN이라 하겠음

path_dict = dir_setting(dir_name, CONTINUE_LEARNING)

model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model)
# CONTINUE_LEARNING가 Ture면 model 재할당