import tensorflow as tf

# ---------- training data로 learning하는 함수 -----------
@tf.function
def go_train(train_ds, model, loss_object, optimizer, metric_objects):

    for images, labels in train_ds:
        # forward propagation
        with tf.GradientTape() as tape:
        # backward propagation에서 계산에 필요한 값을 저장하기 위해 GradientTape 사용
            predictions = model(images)

            loss = loss_object(labels, predictions)
            # CCE 로 loss를 계산

        gradients = tape.gradient(loss, model.trainable_variables)
        # 기울기 계산
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # gradient descent method를 통해 optimizer 

        metric_objects['train_loss'](loss)
       
        # train_loss 안에는 Mean() 이라는 함수가 들어있다. 
        # 즉, Mean(loss) 가 동작하는 code임. cost를 계산.
        metric_objects['train_acc'](labels, predictions)
        # CategoricalAccuracy(labels, predictions)
        

# ---------- validation data로 validation하는 함수 -----------
@tf.function
def go_validation(validation_ds, model, loss_object, metric_objects, con_mat):

    # optimizer 안할거라 Tape 불필요
    for images, labels in validation_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)

        metric_objects['validation_loss'](loss)
        metric_objects['validation_acc'](labels, predictions)

        predictions_argmax = tf.argmax(predictions, axis = 1)
        labels_argmax = tf.argmax(labels, axis = 1)

        con_mat += tf.math.confusion_matrix(labels_argmax, predictions_argmax)


    return con_mat
    
        
        

# ----------test data로 test해보고 그 결과를 저장하는 함수 -----------


def go_test(test_ds, model, loss_object, metric_objects, path_dict, epoch):

    for images, labels in test_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)

        metric_objects['test_loss'](loss)
        metric_objects['test_acc'](labels, predictions)
    
    loss, acc = metric_objects['test_loss'].result().numpy(), metric_objects['test_acc'].result().numpy()
    model_path = path_dict['cp_path']
    if epoch == 0:
        with open(model_path + '/test_result.txt', 'w') as f:
            # 경로 위에 있는 file 열어주고 (경로 위에 file이 없다면 새로 생성)
            f.write(f"    epoch: {epoch} \n")
            f.write(f"test_loss: {loss} \ntest_acc: {acc*100}")
    else :
        with open(model_path + '/test_result.txt', 'a') as f:
            f.write(f"\n\n    epoch: {epoch} \n")
            f.write(f"test_loss: {loss}, \ntest_acc: {acc*100}")
