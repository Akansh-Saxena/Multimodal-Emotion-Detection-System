import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

class GraphGenerator:
    def __init__(self, output_dir="d:/MULTIMODAL_EMOTION_DETECTION_01/outputs/graphs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set visualization style
        sns.set_theme(style="whitegrid", palette="muted")
        
    def plot_training_history(self, history, model_name="Multimodal_Fusion"):
        """Plot Train/Val Loss and Accuracy over Epochs"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(14, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
        plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', marker='x', markersize=4)
        plt.title(f"{model_name} - Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='green', marker='o', markersize=4)
        plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='orange', marker='x', markersize=4)
        plt.title(f"{model_name} - Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        filepath = os.path.join(self.output_dir, f"{model_name}_training_history.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved training history plot to {filepath}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names, model_name="Multimodal_Fusion"):
        """Generate and save visually appealing confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor='gray')
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Emotion")
        plt.ylabel("Actual Emotion")
        
        filepath = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to {filepath}")

    def plot_roc_curve(self, y_true, y_scores, class_names, model_name="Multimodal_Fusion"):
        """Generate one-vs-rest ROC Curves"""
        # Note: y_true should be integer labels. y_scores should be probabilities shape (n_samples, n_classes)
        plt.figure(figsize=(10, 8))
        num_classes = len(class_names)
        
        for i in range(num_classes):
            # Treat class `i` as positive, others as negative
            y_true_binary = (np.array(y_true) == i).astype(int)
            y_score_class = np.array(y_scores)[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_class)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2) # Diagonal random baseline
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        filepath = os.path.join(self.output_dir, f"{model_name}_roc_curve.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved ROC curve plot to {filepath}")


if __name__ == "__main__":
    # Mock Data Generation to test the capability
    generator = GraphGenerator()
    
    # 1. Training History Mock
    mock_history = {
        'train_loss': [1.8, 1.4, 1.1, 0.8, 0.5, 0.3, 0.2],
        'val_loss':   [1.7, 1.5, 1.2, 0.9, 0.7, 0.6, 0.65],
        'train_acc':  [45.0, 58.0, 72.0, 81.0, 88.0, 93.0, 96.0],
        'val_acc':    [48.0, 55.0, 68.0, 78.0, 85.0, 89.0, 91.5]
    }
    generator.plot_training_history(mock_history)
    
    # 2. Confusion Matrix Mock
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    mock_y_true = np.random.randint(0, 7, size=200)
    mock_y_pred = mock_y_true.copy()
    # Introduce some errors for realism
    for _ in range(30):
        idx = np.random.randint(0, 200)
        mock_y_pred[idx] = np.random.randint(0, 7)
    generator.plot_confusion_matrix(mock_y_true, mock_y_pred, classes)
    
    # 3. ROC Curve Mock
    mock_y_scores = np.random.rand(200, 7) 
    # normalize so they sum to 1 to simulate softmax
    mock_y_scores = mock_y_scores / mock_y_scores.sum(axis=1, keepdims=True)
    generator.plot_roc_curve(mock_y_true, mock_y_scores, classes)
    
    print("Graph generation complete. Outputs stored in outputs/graphs/")
