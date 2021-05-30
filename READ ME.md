# READ ME

## Abstract

I Implement Classifier Model  as Convolution Neural Network with Minst from tensorflow dataset and some utility code.

This project is intended for experience about Implement model and utility code like code construction of common thesis  

It shows the changing of loss and accuracy value for every epoch because the code constructed training, validation, test be implemented in same `for loop`.  And the training, validation, test data save in every epoch as a file.

After finish code task, we can check 'confusion matrix'  for last epoch and a plot about changing of loss, accuracy value for full epoch as .png file. 

I also implemented it can continue learning in case after the training stop unintentionally during learning 







## Getting Started

```
MnistClassifier\main.py
```





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

  create directory for save learning data and save the path

- **get_classification_metrics**

  save objects that compute loss, accuracy in the 'metric_objects' dictionary

- **continue_setting**

  if the model's learning is unintentionally interrupted, help it can learning continue 

  

#### 2. dataset_utils.py

- **load_processing_mnist**

  Load Mnist data of tensorflow and do pre-processing



#### 3. train_validation_test.py

- **go_train**

  learning with data for train

- **go_validation**

  validation with data for validation, and return value for confusion_matrix 

- **go_test**

  test with data for test and save the loss and accuracy value as .txt file

  

#### 4. basic_utils.py

- **resetter**

  reset the object in `metric_objects` to initial state in every epoch

- **training_reporter**

  print the loss and accuracy about validation, train data in every epoch

  

#### 5. cp_utils.py

- **save_losses_model**

  save the model learned so far

- **loss_acc_visualizer**

  draw the plot about loss, accuracy of validation by twinx line plot during full epoch and save as .png file

- **confusion_matrix_visualizer**

  draw the confusion_matrix plot and save as .png file

  

## version

| name             | version |
| ---------------- | ------- |
| python           | 3.8.5   |
|                  |         |
| **package name** |         |
| numpy            | 1.19.2  |
| tensorflow       | 2.4.1   |
| matplotlib       | 3.3.2   |









