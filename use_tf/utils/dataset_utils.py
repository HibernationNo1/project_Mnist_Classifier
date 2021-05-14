import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

# data set을 preprocessing하는 함수를 모아놓는 file

def load_processing_mnist(train_ratio, train_batch_size, test_batch_size):
    (train_validation_ds, test_ds), ds_info = tfds.load(name = 'mnist',
                                                        shuffle_files = True,
                                                        as_supervised = True,
                                                        split = ['train', 'test'],
                                                        with_info=True)
    # (train + validation dataset, test dataset), dataset_information

    n_train_validation = ds_info.splits['train'].num_examples  # num_examples : num of dataset
    # dataset_information에 들어있던 train dataset의 개수만 n_train_validation에 할당
    n_train = int(n_train_validation * train_ratio) # train dataset 중에서 실제로 train data로 사용할 비율
    n_validation = n_train_validation - n_train     # 나머지는 validaiton dataset
    # n_train_validation 에서 train, validation을 위한 data 개수를 분류

    train_ds = train_validation_ds.take(n_train) # train_validation_ds에서 n_train 개수만큼 반환
    remaining_ds = train_validation_ds.skip(n_train) # train_validation_ds에서 (전체 - n_train 개수)만큼 반환
    validation_ds = remaining_ds.take(n_validation)

    def normalization(images, labels):
        images = tf.cast(images, tf.float32)/255.

        oh_labels = tf.one_hot(labels, 10)
        return [images, oh_labels]

    train_ds = train_ds.shuffle(1000).map(normalization).batch(train_batch_size)
    
    # ndarray.batch(batch_size) : ndarray를 batch_size만큼씩 잘라서 반환
    # train_ds를 1000개씩 섞고, normalization을 진행한 뒤에, batch_size만큼씩 잘라서 반환
    # data set에 images, labels이란 object가 담겨져 있어 map(normalization) 가능
    validation_ds = validation_ds.map(normalization).batch(test_batch_size)
    test_ds = test_ds.map(normalization).batch(test_batch_size)

    return train_ds, validation_ds, test_ds

def load_processing_cifar10(train_ratio, train_batch_size, test_batch_size):
    (train_validation_ds, test_ds), ds_info = tfds.load(name = 'cifar10',
                                                        shuffle_files = True,
                                                        as_supervised = True,
                                                        split = ['train', 'test'],
                                                        with_info=True)

    n_train_validation = ds_info.splits['train'].num_examples 
    n_train = int(n_train_validation * train_ratio) 
    n_validation = n_train_validation - n_train    

    train_ds = train_validation_ds.take(n_train) 
    remaining_ds = train_validation_ds.skip(n_train) 
    validation_ds = remaining_ds.take(n_validation)

    def normalization(images, labels):
        images = tf.cast(images, tf.float32)/255.
        return [images, labels]

    train_ds = train_ds.shuffle(1000).map(normalization).batch(train_batch_size)
    validation_ds = validation_ds.map(normalization).batch(test_batch_size)
    test_ds = test_ds.map(normalization).batch(test_batch_size)

    return train_ds, validation_ds, test_ds