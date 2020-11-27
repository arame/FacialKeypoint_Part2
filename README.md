# FacialKeypoint_Part2 #

This is the second part of the Facial Keypoint project that I worked on in the Udacity Computer Vision training course.
The code was originally written in Jupyter Notebooks, but I prefer to code in Python files.
This project uses deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. This code can look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.
![alt text](https://github.com/arame/FacialKeypoint_Part2/blob/master/key_pts_example.png?raw=true)

For part 2 we are Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

## Local Environment Instructions ##

Create (and activate) a new environment, named cv-nd with Python 3.6. If prompted to proceed with the install (Proceed [y]/n) type y.

- Linux or Mac:
```
conda create -n cv-nd python=3.6
source activate cv-nd
```
- Windows:
```
conda create --name cv-nd python=3.6
activate cv-nd
```
At this point your command line should look something like: (cv-nd) <User>:P1_Facial_Keypoints <user>$. The (cv-nd) indicates that your environment has been activated, and you can proceed with further package installations.

Install PyTorch and torchvision; this should install the latest version of PyTorch.

- Linux or Mac:
```
conda install pytorch torchvision -c pytorch 
```
- Windows:
```
conda install pytorch-cpu -c pytorch
pip install torchvision
```
Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
## License ##
Open

