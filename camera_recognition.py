# project.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Net(nn.Module):
    '''
    My convolutional neural network!
    '''
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 84, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(84)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(84, 56, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(56)

        self.conv3 = nn.Conv2d(56, 28, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(28)

        self.conv4 = nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(56)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(56, 252)
        self.batchnorm_fc1 = nn.BatchNorm1d(252)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(252, 24)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def load_weights(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def classify(image, net):
    """
    Classify a given image and return the predicted class.
    """
    net.eval()

    # resize and normalize the image
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    
    # convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    # make prediction, apply softmax, get classes index
    index = torch.argmax(net(image_tensor)).item()

    return classes[index]

def run_camera_application(net):
    """
    Run the camera application to classify letters from the webcam feed.
    """
    cap = cv2.VideoCapture(0)

    # check if the webcam is opened successfully
    if not cap.isOpened():
        logger.error("Cannot open camera")
        cap.release()
        cv2.destroyAllWindows()
        return

    while True:
        # capture frame-by-frame
        ret, image = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            logger.error("Can't receive frame (stream end?). Exiting ...")
            break

        # flip the frame horizontally
        image = cv2.flip(image, 1)

        # define the region of interest (ROI)
        top, right, bottom, left = 50, 350, 300, 600
        roi = image[top:bottom, right:left]
        roi = cv2.flip(roi, 1)

        # convert to gray, add Gaussian Blur (makes blurry)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # classifying with CNN
        letter = classify(gray, net)

        # displaying box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN

        # adding prediction to the top of the box
        text_position = (left - 10, top - 10)
        cv2.putText(image, letter, text_position, font, 3, (0, 0, 255), 2)

        # display the resulting frame
        cv2.imshow('image', image)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # create an instance of the Net class
    net = Net()

    # load the weights from the .pt file
    net.load_weights('cnn156.pt')

    run_camera_application(net)