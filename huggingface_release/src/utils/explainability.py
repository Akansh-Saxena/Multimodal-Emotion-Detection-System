import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Explainable AI (XAI) class to visualize what part of an image
    the Convolutional Neural Network is focusing on to make its emotion prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to extract gradients and activations dynamically
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Taking the gradient with respect to the output
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        outputs = self.model(x)
        
        if class_idx is None:
            # If no specific class target, use the highest predicted class
            class_idx = outputs.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        target = outputs[0][class_idx]
        target.backward(retain_graph=True)

        # Get the activations and gradients
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        # Calculate neuron importance weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of forward activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to only keep features that have a positive influence on the class
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def overlay_heatmap(img_path, heatmap_array, output_path="heatmap_output.jpg"):
    """
    Overlays the mathematically derived Grad-CAM heatmap over the original face image.
    """
    # Read original image
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap_array, (img_w, img_h))
    
    # Convert heatmap to RGB 
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine original image with the heatmap
    superimposed_img = heatmap * 0.4 + img * 0.6
    
    cv2.imwrite(output_path, superimposed_img)
    return True

if __name__ == "__main__":
    print("Grad-CAM Explainable AI (XAI) module initialized.")
