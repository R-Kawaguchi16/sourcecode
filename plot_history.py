import matplotlib.pyplot as plt

def plot_history_loss(fit, modelname):
    plt.plot(fit.history['loss'],label="loss for training")
    plt.plot(fit.history['val_loss'],label="loss for validation")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.grid(which="both")
    plt.savefig(modelname+'/loss.eps',
                bbox_inches="tight", pad_inches=0.05)
    plt.savefig(modelname+'/loss.svg',
                bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_history_acc(fit, modelname):
    plt.plot(fit.history['accuracy'],label="accuracy for training")
    plt.plot(fit.history['val_accuracy'],label="accuracy for validation")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.grid(which="both")
    plt.savefig(modelname+'/acc.eps',
                bbox_inches="tight", pad_inches=0.05)
    plt.savefig(modelname+'/acc.svg',
                bbox_inches="tight", pad_inches=0.05)
    plt.close()
