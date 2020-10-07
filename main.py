# import the usual resources
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from models import Net1
from models import Net2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

def main():
    #------------------------------------------------------------------------------------------------------------------
    # Hyperparameters
    hyp_epochs = 5
    hyp_batch_size = 20
    #hyp_optim = "SGD"
    hyp_optim = "Adam"
    #hyp_net = Net1()
    hyp_net = Net2()

    print("Hyperparameters")
    print("--------------")
    print("Epochs = ", hyp_epochs)
    print("Batch Size = ", hyp_batch_size)
    print("Optimizer = ", hyp_optim)
    print("--------------")
    ## TODO: Define the Net in models.py

    net = hyp_net
    print(net)

    ## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
    # order matters! i.e. rescaling should come before a smaller crop
    data_transform = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        Normalize(),
        ToTensor()
    ])
    # testing that you've defined a transform
    assert(data_transform is not None), 'Define a data_transform'

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file='../files/training_frames_keypoints.csv',
                                                root_dir='../files/training/',
                                                transform=data_transform)


    print('Number of images: ', len(transformed_dataset))

    # iterate through the transformed dataset and print some stats about the first few samples
    for i in range(4):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())
        
    # load training data in batches
    batch_size = hyp_batch_size

    train_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0)
                            
    # load in the test data, using the dataset class
    # AND apply the data_transform you defined above

    # create the test dataset
    test_dataset = FacialKeypointsDataset(csv_file='../files/test_frames_keypoints.csv',
                                                root_dir='../files/test/',
                                                transform=data_transform)
    # load test data in batches
    batch_size = hyp_batch_size

    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0)

    # test the model on a batch of test images
    # call the above function
    # returns: test images, test predicted keypoints, test ground truth keypoints
    test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)

    # print out the dimensions of the data to see if they make sense
    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())
    # visualize the output
    # by default this shows a batch of 10 images
    # call it
    _visualise = False
    if _visualise == True:
        visualize_output(test_images, test_outputs, gt_pts)

    ## TODO: Define the loss and optimization
    import torch.optim as optim

    criterion = nn.MSELoss()

    hyp_optimizer = None
    if hyp_optim == "Adam":
        hyp_optimizer = optim.Adam(net.parameters(), lr=0.001)

    if hyp_optim == "SGD":
        hyp_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    optimizer = hyp_optimizer
    # train your network
    n_epochs = hyp_epochs # start small, and increase when you've decided on your model structure and hyperparams

    # this is a Workspaces-specific context manager to keep the connection
    # alive while training your model, not part of pytorch
    train_net(n_epochs, train_loader, net, criterion, optimizer)
        
    # get a sample of test data again
    test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())

    ## TODO: change the name to something uniqe for each new model
    model_dir = '../saved_models/'
    model_name = 'keypoints_model_2.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)
    # --------------------------------------------------------------------
    # To run the following code after retreiving an existing model, 
    # you can do so in the resume.py file
    # --------------------------------------------------------------------

    # Get the weights in the first conv layer, "conv1"
    # if necessary, change this to reflect the name of your first conv layer
    weights1 = net.conv1.weight.data

    w = weights1.numpy()

    filter_index = 0

    print(w[filter_index][0])
    print(w[filter_index][0].shape)

    # display the filter weights
    plt.imshow(w[filter_index][0], cmap='gray')

    ##TODO: load in and display any image from the transformed test dataset
    i = 1
    show_image(test_images, w, i)
    ## TODO: Using cv's filter2D function,
    ## apply a specific set of filter weights (like the one displayed above) to the test image

    # END OF MAIN =======================================================================================

def net_sample_output(test_loader, net):
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts
	
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()

def show_image(test_images, weights, i):
    plt.figure(figsize=(10, 5))
    # un-transform the image data
    image = test_images[i].data   # get the image from it's Variable wrapper
    image = image.numpy()   # convert to numpy array from a Tensor
    image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image
    filter_index = 0
    dst = cv2.cv2.filter2D(image, -1, weights[filter_index][0])
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    plt.show()

def train_net(n_epochs, train_loader, net, criterion, optimizer):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':    
    main()

