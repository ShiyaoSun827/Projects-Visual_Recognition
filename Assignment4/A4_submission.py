import numpy as np

import torch
from PIL import Image

from torch.utils.data.dataset import Dataset  
from torchvision import transforms
import torchvision

from PIL import Image
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.io import imread

import matplotlib.pyplot as plt
import os

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Contracting Path
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)#input in channels is 3.
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
         # Expansive Path
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
         # Contracting Path
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # Expansive Path
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            #Sine(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            #Sine(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand
    


class validDataset(Dataset):
    def __init__(self, images):

        self.images = images

        self.transforms_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, idx):
        image = self.images[idx].reshape(64, 64, 3)
        trans_image = self.transforms_image(image)
        return trans_image
def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)

    # add your code here to fill in pred_class and pred_bboxes
    images = torch.from_numpy(images)
    images = torch.squeeze(images)
    unflatten1 = torch.nn.Unflatten(dim=1, unflattened_size=(3, 64, 64))
    images = unflatten1(images)
    images = images.detach().cpu().numpy()
    model = torch.hub.load('ultralytics/yolov5', 'custom',path='./best.pt',force_reload=True, trust_repo = True)

    if torch.cuda.is_available():
        model = model.to('cuda')


    model.conf = 0.23
    model.multi_label = True
    model.agnostic = False

    for i in range(len(images)):#for every image
        np_image = images[i]
        np_image = np_image.reshape((64, 64, 3))#reshape the image
        im = Image.fromarray(np_image, 'RGB')
        results = model(im, size=640)

        '''
        one sample results.xyxy[0] of YOLOV5 looks like:
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
        # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
        '''
        if len(results.xyxy[0]) < 2:#In this case only one class(two same number)
            '''
            tensor([[34.09883, 19.02788, 62.16243, 46.89036,  0.95352,  9.00000]], device='cuda:0')
            image_name:194

            '''
            pred_class[i][0:2] = [results.xyxy[0][0][5].cpu(), results.xyxy[0][0][5].cpu()]#we need to add the class to predict
            #we need ymin xmin ymax xmax, result.xyxy is xmin,ymin,xmax,ymax
            pred_bboxes[i][0][0] = results.xyxy[0][0][1].cpu()
            pred_bboxes[i][0][1] = results.xyxy[0][0][0].cpu()
            pred_bboxes[i][0][2] = results.xyxy[0][0][3].cpu()
            pred_bboxes[i][0][3] = results.xyxy[0][0][2].cpu()



            #Same as the second number
            pred_bboxes[i][1][0] = results.xyxy[0][0][1].cpu()
            pred_bboxes[i][1][1] = results.xyxy[0][0][0].cpu()
            pred_bboxes[i][1][2] = results.xyxy[0][0][3].cpu()
            pred_bboxes[i][1][3] = results.xyxy[0][0][2].cpu()


        else:

            '''
            about image.xyxy[0] with two different number:
            ################
            tensor([[22.95975, 31.05684, 50.86736, 58.98111,  0.96705,  3.00000],
            [35.68084, 33.08403, 63.66699, 61.36357,  0.96705,  5.00000]], device='cuda:0')
            ################

            '''
            if results.xyxy[0][0][5] > results.xyxy[0][1][5]:#compare the number,if the 1st > the 2nd
                pred_class[i][0] = results.xyxy[0][1][5].cpu()
                pred_class[i][1] = results.xyxy[0][0][5].cpu()
                #add bboxes
                pred_bboxes[i][0][0] = results.xyxy[0][1][1].cpu()
                pred_bboxes[i][0][1] = results.xyxy[0][1][0].cpu()
                pred_bboxes[i][0][2] = results.xyxy[0][1][3].cpu()
                pred_bboxes[i][0][3] = results.xyxy[0][1][2].cpu()


                #add the another one
                pred_bboxes[i][1][0] = results.xyxy[0][0][1].cpu()
                pred_bboxes[i][1][1] = results.xyxy[0][0][0].cpu()
                pred_bboxes[i][1][2] = results.xyxy[0][0][3].cpu()
                pred_bboxes[i][1][3] = results.xyxy[0][0][2].cpu()

            else:
                #if 1st < 2nd
                pred_class[i][0] = results.xyxy[0][0][5].cpu()
                pred_class[i][1] = results.xyxy[0][1][5].cpu()

                #add bboxes
                pred_bboxes[i][0][0] = results.xyxy[0][0][1].cpu()
                pred_bboxes[i][0][1] = results.xyxy[0][0][0].cpu()
                pred_bboxes[i][0][2] = results.xyxy[0][0][3].cpu()
                pred_bboxes[i][0][3] = results.xyxy[0][0][2].cpu()

                #add the another one
                pred_bboxes[i][1][0] = results.xyxy[0][1][1].cpu()
                pred_bboxes[i][1][1] = results.xyxy[0][1][0].cpu()
                pred_bboxes[i][1][2] = results.xyxy[0][1][3].cpu()
                pred_bboxes[i][1][3] = results.xyxy[0][1][2].cpu()
    pred_bboxes = np.round(pred_bboxes)


    dataset = validDataset(images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    N = images.shape[0]
    network = UNET(3,11).to(device)
    checkpoint = torch.load('./Unet.pt')
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    all_predictions = []
    with torch.no_grad():
      for _, images in enumerate(loader):
        data = images.to(device)
        output = network(data)
        all_predictions.append(output.cpu())
    final_predictions = torch.cat(all_predictions, dim=0)
    pred_seg = torch.argmax(final_predictions, dim=1)
    pred_seg = pred_seg.view(pred_seg.size(0), -1).cpu().numpy()
    


    return pred_class, pred_bboxes, pred_seg
