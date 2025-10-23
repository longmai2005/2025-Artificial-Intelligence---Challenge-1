import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

CM_PATH = "confusion_matrix.csv"
CLASSES_PATH = "classes.txt"
OUTPUT_PATH = "cm.png"

def load_confusion_matrix():
    """Load confusion matrix from CSV"""
    data = []
    with open(CM_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            
            values = [int(x) for x in parts[1:]]
            data.append(values)
    return np.array(data)

def load_classes():
    """Load class names"""
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]
    
def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title="Confusion Matrix",
           ylabel="True label", 
           xlabel="Predicted label")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") 
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize = 14, fontweight='bold')
    fig.tight_layout()
    return fig

def main(): 
    print("=" * 60)
    print("CONFUSION MATRIX VISUALIZATION")
    print("=" * 60)
    
    print(f"\nLoading confusion matrix from: {CM_PATH}...")
    cm = load_confusion_matrix()
    
    print(f"Loading classes from: {CLASSES_PATH}...")
    classes = load_classes()
    
    print(f"\nConfusion matrix shape: {cm.shape}")
    print(f"Classes: {classes}")
    
    print("\nGenerating confusion matrix plot...")
    fig = plot_confusion_matrix(cm, classes)
    
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix plot saved to: {OUTPUT_PATH}")
    
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED!")
    print("=" * 60)
    
if __name__ == "__main__":
    main()