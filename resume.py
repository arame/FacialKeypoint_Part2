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
    hyp_batch_size = 20
    net = Net2()
    model_dir = '../saved_models/'
    model_name = 'keypoints_model_2.pt'
    data_transform = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        Normalize(),
        ToTensor()
    ])

    # retreive the saved model
    net_state_dict = torch.load(model_dir+model_name)
    net.load_state_dict(net_state_dict)
    
    # load the test data
    test_dataset = FacialKeypointsDataset(csv_file='../files/test_frames_keypoints.csv',
                                                root_dir='../files/test/',
                                                transform=data_transform)
    # load test data in batches
    batch_size = hyp_batch_size

    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0)

    test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())
    # Get the weights in the first conv layer, "conv1"
    # if necessary, change this to reflect the name of your first conv layer
    weights1 = net.conv1.weight.data

    w = weights1.numpy()

    filter_index = 0

    print(w[filter_index][0])
    print(w[filter_index][0].shape)

    # display the filter weights
    plt.imshow(w[filter_index][0], cmap='gray')
    #plt.show()
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


def show_image(test_images, weights, i):
    plt.figure(figsize=(10,5))
    # un-transform the image data
    image = test_images[i].data   # get the image from it's Variable wrapper
    image = image.numpy()   # convert to numpy array from a Tensor
    image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image
    filter_index = 0
    dst = cv2.filter2D(image, -1, weights[filter_index, 0])
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':    
    main()