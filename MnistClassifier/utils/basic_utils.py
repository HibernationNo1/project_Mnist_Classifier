# 각각의 object에 찌꺼기가 남아있는 것을 방지
def resetter(metric_objects):
    metric_objects['train_loss'].reset_states()
    metric_objects['train_acc'].reset_states()
    metric_objects['validation_loss'].reset_states()
    metric_objects['validation_acc'].reset_states()
    # reset_states() : initial 상태로 되돌림

# training data를 learning하며 얻어진 value들을 쌓아가며 기록하는 함수
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
        # dir_name은 model의 name. 
        # 다수의 model을 돌릴때, 한 개의 model에 대한 learning이 끝나면
        # 다음 순서의 model을 learning하기 때문에 
        # 지금 무슨 model을 돌리고 있는지 확인하기 위해 쓴 code
        print(dir_name)
    print(epoch)

    print(f"Train Loss: {losses_accs['train_losses'][-1]:.4f}")
    print(f"Train Accuracy: {losses_accs['train_accs'][-1]:.2f}")
    print(f"Validation Loss: {losses_accs['validation_losses'][-1]:.4f}")
    print(f"Validation Accuracy: {losses_accs['validation_accs'][-1]:.2f}")