import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data_loaders
from models import FCNet, ConvNet
import matplotlib.pyplot as plt

### Fixed variables ###
BATCH_SIZE = 1000
SEED = 42
#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setup the device for tensor to be stored

### Setup seeds for deterministic behaviour of computations ###
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
###############################################################

# Load dataset
CIFAR_10_dataset = get_data_loaders(BATCH_SIZE)
    
def train_and_test(model_name, dataset, num_epochs, learning_rate, activation_function_name):
    if model_name == "fcnet":
        model = FCNet(activation_function_name=activation_function_name).to(device)
    elif model_name == "convnet":
        model = ConvNet(activation_function_name=activation_function_name).to(device)
    else:
        raise Exception("No such model. The options are 'fcnet' and 'convnet'.")
    train_loader = dataset[0]  # Stores both the training data and the label
    test_loader = dataset[1]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    test_accuracy = None
    
    # Train the model here: use functions in the nn.Module PyTorch library
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for train_image, train_label in train_loader:
            train_image, train_label = train_image.to(device), train_label.to(device)
            
            optimizer.zero_grad()  # Clear out all the previous gradients
            predicted_label = model(train_image) # Forward passing the images to get predicted labels
            loss = criterion(predicted_label, train_label)  # Calculates the loss for the predicted vs actual labels
            loss.backward()  # Backward propagation 
            optimizer.step()  # Update the weights based on the backpropagation
            running_loss += loss.item()
    
    # Test the model here (return the model and test accuracy in percentage)
    model.eval()  # Set the "evaluation mode" for the model
    num_correct, num_total = 0.0, 0.0
    with torch.no_grad():
        for test_image, test_label in test_loader:
            test_image, test_label = test_image.to(device), test_label.to(device)
            num_total += test_label.size(0)

            classification_vector = model(test_image)  
            _, predicted_label = torch.max(classification_vector, 1)  # Chooses the label with the highest score/the predicted classification
            num_correct += (predicted_label == test_label).sum().item()
    
    test_accuracy = (num_correct / num_total) * 100
    return model, test_accuracy
    

def hyperparameters_grid_search(model_name, dataset):
    learning_rate_options = [1e-7, 1e-3, 1]
    activation_function_name_options = ["sigmoid", "relu"]
    best_test_accuracy = 0
    best_hyperparameters = {"learning_rate": None, "activation_function_name": None}
    # TODO: Complete grid search on learning rates and activation functions. Keep the number of epochs to be 5.
    # You can use the following print statements to keep track of the hyperparameter search and finally output the best hyperparameters as well as the 
    # print(f"Current hyperparameters: num_epochs=5, learning_rate={_}, activation_function_name={_}")
    # print(f"Current accuracy for test images: {_}%")
    # print(f"Best test accuracy: {best_test_accuracy}%")
    # print("Best hyperparameters:", best_hyperparameters)
    

if __name__ == "__main__":
    # Train and test fully-connected neural network and save the model
    model, test_accuracy = train_and_test(model_name="fcnet", dataset=CIFAR_10_dataset, num_epochs=5, learning_rate=1e-3, activation_function_name="relu")
    print(f"FCN Test accuracy is: {test_accuracy}%")
    torch.save(model.state_dict(), "cifar-10-fcn.pt")
    print("FCN model saved.")

    # Plot images with predictions
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    images, labels = next(iter(CIFAR_10_dataset[1])) 
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    fig = plt.figure("Example Predictions", figsize=(12, 5))
    fig.suptitle("Example Predictions")
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        img = images[i] * 0.25 + 0.5     # unnormalize
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"Label: {classes[labels[i]]}\nPredicted: {classes[predicted[i]]}")
    plt.show()

    # Train and test convolutional neural network and save the model
    model, test_accuracy = train_and_test(model_name="convnet", dataset=CIFAR_10_dataset, num_epochs=5, learning_rate=1e-3, activation_function_name="relu")
    print(f"CNN Test accuracy is: {test_accuracy}%")
    torch.save(model.state_dict(), "cifar-10-cnn.pt")
    print("CNN model saved.")

    # Do hyperparameter search
    hyperparameters_grid_search("convnet", CIFAR_10_dataset)
