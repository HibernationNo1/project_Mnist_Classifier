# READ ME

## CNN Classifier with Mnist

Classifier Model Implementation as Convolution Neural Network





## Project Configuration

- Utility Method

  Configured five utility python code file

  1. learning_env_setting.py
  2. dataset_utils.py
  3. train_validation_test.py
  4. basic_utils.py
  5. cp_utils.py

  Please read the [Method Description](#method-description) for a detailed description 

- Classifier Model

  Implementation using keras

  



## Method Description

#### 1. learning_env_setting.py

- **dir_setting**

  project에 관한 directory를 생성하고 path를 dictionary에 저장

- **get_classification_metrics**

  loss와 accuracy를 계산할 tensorflow.keras의 Mean, CategoricalAccuracy를 'metric_objects' dictionary에 저장

- **continue_setting**

  model의 학습이 의도치 않게 멈췄을 때 저장된 model data를 load해서 이어서 학습하게 해주는 method

  

#### 2. dataset_utils.py

- **load_processing_mnist**

  tensorflow의 Mnist data을 load하고 processing을 진행



#### 3. train_validation_test.py

- **go_train**

  training data로 learning

- **go_validation**

  validation data로 learning을 진행하고 confusion_matrix를 return

- **go_test**

  test data로 learning을 진행하고 loss와 accuracy를 .txt file로 저장

  



#### 4. basic_utils.py

- **resetter**

  metric_objects 안의 keras.Mean, CategoricalAccuracy 값들을 초기 상태로 reset

- **training_reporter**

  매 epoch마다 validation, train data에 의한 loss와 accuracy를 print

  

#### 5. cp_utils.py

- **save_losses_model**

  이제까지 학습 된 model을 저장

- **loss_acc_visualizer**

  validation, train data에 의한 loss와 accuracy를 twinx line plot으로 표현 후 .png file로 save

- **confusion_matrix_visualizer**

  confusion_matrix 값으로 matshow plot을 표현 후 .png file로 save

  



## Getting Started







## version

| name             | version |
| ---------------- | ------- |
| python           | 3.8.5   |
|                  |         |
| **package name** |         |
| numpy            | 1.19.2  |
| tensorflow       | 2.4.1   |
| matplotlib       | 3.3.2   |









