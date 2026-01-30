import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# same cnn architecture (must match training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# cifar10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def load_model():
    """load the trained model from .pth file"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # load the saved weights into the model
    model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=device))
    model.eval()  # set to evaluation mode
    
    print(f"model loaded on {device}")
    return model, device

def get_test_images(num_images=10):
    """get some test images from cifar10"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=num_images, shuffle=True)
    
    # get one batch of images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    return images, labels

def predict_images(model, images, device):
    """run predictions on images"""
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        # get the class with highest probability
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu()

def show_predictions(images, true_labels, predicted_labels):
    """visualize images with predictions"""
    num_images = len(images)
    
    # create grid layout (2 rows of 5 for 10 images)
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    
    for idx, ax in enumerate(axes.flat):
        if idx >= num_images:
            ax.axis('off')
            continue
            
        # unnormalize image for display
        img = images[idx].numpy()
        img = img * np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        img = img + np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        ax.imshow(img)
        
        true_class = classes[true_labels[idx]]
        pred_class = classes[predicted_labels[idx]]
        
        # color: green if correct, red if wrong
        color = 'green' if true_labels[idx] == predicted_labels[idx] else 'red'
        
        ax.set_title(f'true: {true_class}\npred: {pred_class}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("predictions saved to predictions.png")
    plt.show()

def main():
    print("=" * 50)
    print("CIFAR-10 CNN Classifier - Prediction Demo")
    print("=" * 50)
    
    print("\nloading trained model...")
    model, device = load_model()
    
    print("getting 10 random test images...")
    images, true_labels = get_test_images(num_images=10)
    
    print("making predictions...")
    predicted_labels = predict_images(model, images, device)
    
    print("\n" + "-" * 50)
    print("RESULTS:")
    print("-" * 50)
    for i in range(len(true_labels)):
        true_class = classes[true_labels[i]]
        pred_class = classes[predicted_labels[i]]
        correct = "✓" if true_labels[i] == predicted_labels[i] else "✗"
        print(f"{correct} image {i+1:2d}: true={true_class:6s}, predicted={pred_class:6s}")
    
    accuracy = (predicted_labels == true_labels).sum().item() / len(true_labels)
    print("-" * 50)
    print(f"accuracy on these {len(true_labels)} images: {accuracy*100:.1f}%")
    print("-" * 50)
    
    print("\nvisualizing predictions...")
    show_predictions(images, true_labels, predicted_labels)

if __name__ == '__main__':
    main()
