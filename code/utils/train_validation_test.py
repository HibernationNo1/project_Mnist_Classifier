import tensorflow as tf

@tf.function
def go_train(train_ds, model, loss_object, optimizer, metric_objects):
    for images, labels in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(images)

            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_objects['train_loss'](loss)
        metric_objects['train_acc'](labels, predictions)
        
@tf.function
def go_validation(validation_ds, model, loss_object, metric_objects, con_mat):
    for images, labels in validation_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)

        metric_objects['validation_loss'](loss)
        metric_objects['validation_acc'](labels, predictions)

        predictions_argmax = tf.argmax(predictions, axis = 1)
        labels_argmax = tf.argmax(labels, axis = 1)

        con_mat += tf.math.confusion_matrix(labels_argmax, predictions_argmax)


    return con_mat
    
        

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
            f.write(f"    epoch: {epoch} \n")
            f.write(f"test_loss: {loss} \ntest_acc: {acc*100}")
    else :
        with open(model_path + '/test_result.txt', 'a') as f:
            f.write(f"\n\n    epoch: {epoch} \n")
            f.write(f"test_loss: {loss}, \ntest_acc: {acc*100}")
