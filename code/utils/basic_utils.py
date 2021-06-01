def resetter(metric_objects):
    metric_objects['train_loss'].reset_states()
    metric_objects['train_acc'].reset_states()
    metric_objects['validation_loss'].reset_states()
    metric_objects['validation_acc'].reset_states()

def training_reporter(epoch, losses_accs, metric_objects, dir_name = None):
    train_loss = metric_objects['train_loss']
    train_acc = metric_objects['train_acc']
    validation_loss = metric_objects['validation_loss']
    validation_acc = metric_objects['validation_acc']

    losses_accs['train_losses'].append(train_loss.result().numpy())
    losses_accs['train_accs'].append(train_acc.result().numpy()*100)
    losses_accs['validation_losses'].append(validation_loss.result().numpy())
    losses_accs['validation_accs'].append(validation_acc.result().numpy()*100)

    if dir_name:

        print(dir_name)
    print(epoch)

    print(f"Train Loss: {losses_accs['train_losses'][-1]:.4f}")
    print(f"Train Accuracy: {losses_accs['train_accs'][-1]:.2f}")
    print(f"Validation Loss: {losses_accs['validation_losses'][-1]:.4f}")
    print(f"Validation Accuracy: {losses_accs['validation_accs'][-1]:.2f}")