import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def hist_class_distribution(set_x, ax, commands):
    frequencies = []
    
    for command in commands:
        frequencies.append(set_x[set_x.label==command].shape[0])
        
    frequencies = np.array(frequencies)/len(set_x)
    
    ax.bar(commands, frequencies, edgecolor='black', alpha=0.5, color='forestgreen')
    
    
def plot_spectrogram(spectrogram, ax):
  
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)

    # Convert the frequencies to log scale and transpose, so that the time is represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
    
    
def plot_mfcc(features, title=None):

    _, ax = plt.subplots(figsize=(14, 10))
    ax = sns.heatmap(features)
    ax.set_xlabel("Window", fontsize=18)
    ax.set_ylabel("Features", fontsize=18)

    if title is not None:
        ax.set_title(title, fontsize=20)
            
    plt.tight_layout()
    plt.show()
    
    
def plot_history(history, columns=['loss']):
    
    _, axes = plt.subplots(len(columns), 1, figsize=(8, 5*len(columns)))

    for i, column in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes
        ax.plot(history.history[column], label='training', color='blue', linewidth=1.5)
        ax.plot(history.history['val_'+column], label='validation', color='firebrick', linewidth=1.5)
        ax.set_xticks(range(len(history.history['loss'])), labels=range(1, len(history.history['loss'])+1))
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.5)
        ax.set_ylabel(column)
        ax.legend(edgecolor='black', facecolor='linen', fontsize=12, loc ='best') 

    plt.tight_layout()
    plt.show()
    

def plot_confusion_matrix(cm, labels, annot=True, cmap='Blues'):

    _, ax = plt.subplots(figsize=(12, 9))
    ax = sns.heatmap(cm, annot=annot, xticklabels=labels, yticklabels=labels, cmap=cmap)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')

    plt.tight_layout()
    plt.show()
    
    