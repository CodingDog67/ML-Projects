import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_train_test_loss(train_losses, test_losses):
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.title('Model training loss plot')
    plt.legend()
    plt.show()

def plot_train_test_acc(train_acc, test_acc):
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.title('Model training acc plot')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False,  cmap=plt.cm.Blues):
    
    if normalize:
        print('Normalized confusion matrix')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion Matrix not normalizs')
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt), horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()