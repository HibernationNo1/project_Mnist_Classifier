import os
import numpy as np
import matplotlib.pyplot as plt

def save_losses_model(epoch, model, losses_accs, path_dict):
    save_path = os.path.join(path_dict['model_path'], 'epoch_' + str(epoch))
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    
    np.savez_compressed(os.path.join(save_path, 'losses_accs'),
                        train_losses = losses_accs['train_losses'], 
                        train_accs = losses_accs['train_accs'],
                        validation_losses = losses_accs['validation_losses'],
                        validation_accs = losses_accs['validation_accs'])

def loss_acc_visualizer(epoch, losses_accs, path_dict):
    load_path = os.path.join(path_dict['model_path'], 'epoch_' + str(epoch))
    losses_accs = np.load(load_path +'/losses_accs.npz')

    fig, ax_loss = plt.subplots(figsize = (21, 9))
    ax2 = ax_loss.twinx()

    epoch_range = np.arange(1, 1+len(losses_accs['train_losses'])) 

    ax_loss.plot(epoch_range, losses_accs['train_losses'], color = 'tab:blue', 
            linestyle = ':', linewidth = 2, label = 'Train Loss')
    ax_loss.plot(epoch_range, losses_accs['validation_losses'], color = 'tab:blue', 
                linewidth = 2, label = 'validation Loss')

    ax2.plot(epoch_range, losses_accs['train_accs'], color = 'tab:orange', 
            linestyle = ':', linewidth = 2, label = 'Train Accuracy')
    ax2.plot(epoch_range, losses_accs['validation_accs'], color = 'tab:orange', 
            linewidth = 2, label = 'Validation Accuracy')
    
    ax_loss.legend(bbox_to_anchor = (1, 0.5), loc = 'upper right', 
                fontsize = 20, frameon = False)
    ax2.legend(bbox_to_anchor = (1, 0.5), loc = 'lower right', 
                fontsize = 20, frameon = False)

    ax_loss_yticks = ax_loss.get_yticks()
    ax2_yticks = ax2.get_yticks()

    ax_loss_yticks_M = ax_loss_yticks[-1]

    ax_loss_yticks = np.linspace(0, ax_loss_yticks_M, 7)
    ax2_yticks = np.arange(70, 101, 5)
    ax2_yticks_minor = np.arange(70, 101, 1)

    ax_loss.set_yticks(ax_loss_yticks)
    ax_loss.set_ylim([0, ax_loss_yticks_M])
    ax_loss.set_yticklabels(np.around(ax_loss_yticks, 2))
    ax2.set_ylim([70, 100])
    ax2.set_yticks(ax2_yticks)
    ax2.set_yticks(ax2_yticks_minor, minor = True)

    epoch_ticks = np.linspace(1, len(losses_accs['train_losses']), 10).astype(np.int)

    ax_loss.tick_params(labelsize = 20, color = 'tab:blue')
    ax2.tick_params(labelsize = 20, color = 'tab:orange' )
    ax2.tick_params(which = 'minor', right = False)

    ax_loss.set_xticks(epoch_ticks)
    ax2.set_xticks(epoch_ticks)
    ax_loss.set_xticklabels(epoch_ticks, color = 'k')

    ax2.grid(axis ='y')
    ax2.grid(which = 'minor', linestyle = ':')

    ax_loss.set_xlim([1, len(losses_accs['train_losses'])])
    ax2.set_xlim([1, len(losses_accs['train_losses'])])

    ax_loss.set_ylabel('Cross Entropy Loss', fontsize = 30, color = 'tab:blue')
    ax2.set_ylabel('Accuracy', fontsize = 30, color = 'tab:orange')
    fig.tight_layout(pad = 5)


    save_path = os.path.join(path_dict['model_path'], 'epoch_' + str(epoch))
    plt.savefig(save_path + '/losses_accs_visualization.png')

    plt.close()

def confusion_matrix_visualizer(con_mat, n_class, path_dict, epoch):
    fig, ax = plt.subplots(figsize = (14, 14))
    ax.matshow(con_mat, cmap = 'Reds')
    M = np.max(con_mat)
    for r_idx in range(con_mat.shape[0]):
        for c_idx in range(con_mat.shape[1]):
            if con_mat[r_idx, c_idx] > M*0.5:
                color = 'w'
            else:
                color = 'k'
            ax.text(x = c_idx, y = r_idx, s = con_mat[r_idx, c_idx].numpy(), 
                    fontsize = 15,
                    ha = 'center', va = 'center', 
                    color= color)
    
    for  spine in ax.spines.values():
        spine.set_visible(False)
    ticks = np.arange(n_class)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.tick_params(left = False, top = False, bottom = False,
                    labeltop = False, labelbottom = True, 
                    labelsize = 20)
    ax.tick_params(colors = 'royalblue')

    ax.set_title('Confusion Matrix for Validation DS', fontsize = 40, color = 'royalblue')
    ax.set_ylabel('True Labels', fontsize = 30, color = 'royalblue')
    ax.set_xlabel('Predicted Label', fontsize = 30, color = 'royalblue')

    fig.tight_layout()
    save_path = os.path.join(path_dict['model_path'], 'epoch_' + str(epoch))
    plt.savefig(save_path + '/confustion_matrix_visualization.png')
    
    plt.close()
