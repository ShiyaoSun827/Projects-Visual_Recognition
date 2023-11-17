from ultralytics import YOLO
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, gt_classes, gt_bboxes,gt_seg):
        assert images.shape[0] == gt_classes.shape[0]
        self.images = images
        self.gt_classes = gt_classes
        self.bboxes = gt_bboxes
        self.gt_seg = gt_seg
        self.transforms_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

        ])


     
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(64,64,3)
        gt_class = self.gt_classes[idx]
        gt_bbox = self.bboxes[idx]
        gt_segm = self.gt_seg[idx].reshape(64, 64)

        t_image = self.transforms_image(image)
        gt_class = torch.tensor(gt_class,dtype=torch.float32)
        gt_bbox = torch.tensor(gt_bbox,dtype=torch.float64)
        gt_segm = torch.tensor(gt_segm,dtype=torch.float64)

        return t_image,gt_class,gt_bbox,gt_segm
def save_image():
    images =np.load("valid_X.npy")
    '''
    dataset_folder = "dataset"
    images_folder = "images"
    labels_folder = "labels"
    os.makedirs(f"{dataset_folder}/{images_folder}", exist_ok=True)
    os.makedirs(f"{dataset_folder}/{labels_folder}", exist_ok=True)
    '''
    index = 0
    for i in images:
        image_filename = '{}.png'.format(index)
        image = i.reshape(64, 64, 3)
        #image_path = os.path.join(dataset_folder, images_folder, image_filename)
        
        image_path = os.path.join("dataset\images_val", image_filename)
        cv2.imwrite(image_path, image)
        index += 1


def extract_info():
    images =np.load("train_X.npy")
    gt_classes = np.load('valid_Y.npy')
    gt_bboxes = np.load('valid_bboxes.npy')
    gt_seg = np.load('train_seg.npy')
    info_dict = {}
    info_dict['bboxes'] = []
    
    #deal with bboxes
    bbox = {}
    temp = []
    for i in gt_classes:
        for j in i:

            bbox["class"] = str(j)
            temp.append(bbox)
            bbox = {}
    filename = 1
    outbut = []
    index = 0
    for sub in gt_bboxes:
        for k in range(4):
            if k == 0:
                temp[index]["ymin"] = sub[0][k]
            elif k == 1:
                temp[index]["xmin"] = sub[0][k]
            elif k == 2:
                temp[index]["ymax"] = sub[0][k]
            elif k ==3:
                temp[index]["xmax"] = sub[0][k]
        index+=1
        for k in range(4):
            if k == 0:
                temp[index]["ymin"] = sub[1][k]
            elif k == 1:
                temp[index]["xmin"] = sub[1][k]
            elif k == 2:
                temp[index]["ymax"] = sub[1][k]
            elif k ==3:
                temp[index]["xmax"] = sub[1][k]
        index+=1
        filename += 1
   
    name = 0
    temp2 = []
    for i in range(10000):
        temp2.append(temp[i])
        if i %2 == 1:
            if len(temp2) != 0:
                tempdict = {}
                tempdict["bboxes"] = temp2
                tempdict["filename"] = "{}.png".format(name)
                outbut.append(tempdict)
                name = name + 1

            temp2 = []
        
        
        


    
    return outbut
    
class_name = {"0":0,
                "1":1,
                "2":2,
                "3":3,
                "4":4,
                "5":5,
                "6":6,
                "7":7,
                "8":8,
                "9":9
                
                }
def convert_toyolo5(list):
    # Get the annotations
    annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]
    annotations.sort()
    for input_dict in list:
        print_buffer =[]
        for b in input_dict["bboxes"]:
            try:
                class_id = class_name[b["class"]]
            except KeyError:
                print("!!!")
            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (b["xmin"] + b["xmax"]) / 2 
            b_center_y = (b["ymin"] + b["ymax"]) / 2
            b_width    = (b["xmax"] - b["xmin"])
            b_height   = (b["ymax"] - b["ymin"])
            
            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = 64,64,3
            b_center_x /= image_w 
            b_center_y /= image_h 
            b_width    /= image_w 
            b_height   /= image_h 
            
            #Write the bbox details to the file 
            print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
            
        # Name of the file which we have to save 
        save_file_name = os.path.join("annotations", input_dict["filename"].replace("png", "txt"))
        
        # Save the annotation to disk
        print("\n".join(print_buffer), file= open(save_file_name, "w"))
    annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
    return annotations



    

 



def main():
    # Load a model

    '''
    images =np.load("train_X.npy")
    gt_classes = np.load('train_Y.npy')
    gt_bboxes = np.load('train_bboxes.npy')
    gt_seg = np.load('train_seg.npy')
    train_dataset = CustomDataset(images, gt_classes, gt_bboxes, gt_seg)
    train_loader = DataLoader(train_dataset,batch_size=10,shuffle=True)
    '''
    '''
    import os
    import cv2

    dataset_folder = "dataset"
    images_folder = "images"
    labels_folder = "labels"
    os.makedirs(f"{dataset_folder}/{images_folder}", exist_ok=True)
    os.makedirs(f"{dataset_folder}/{labels_folder}", exist_ok=True)

    for i, (image, gt_classes, gt_bboxes, _) in enumerate(train_dataset):
        # Save the image
        image_filename = f"{i}.jpg"
        image_path = os.path.join(dataset_folder, images_folder, image_filename)
        cv2.imwrite(image_path, image[:, :, [2, 1, 0]])  # Convert from RGB to BGR

        # Save the bounding box annotations
        label_filename = f"{i}.txt"
        label_path = os.path.join(dataset_folder, labels_folder, label_filename)
        with open(label_path, 'w') as f:
            for bbox, gt_class in zip(gt_bboxes, gt_classes):
                # Convert from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                # Normalize to [0, 1]
                x_center /= 64
                y_center /= 64
                width /= 64
                height /= 64
                # Write to file
                f.write(f"{int(gt_class)} {x_center} {y_center} {width} {height}\n")



'''



    model = YOLO('mymodel.yaml')  # build a new model from YAML
   
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data='data.yaml',epochs=50,batch = -1, imgsz=128)
    #results = model.train(data=train_loader,epochs=100, imgsz=640)
    
def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

if __name__ == '__main__':
    main()
    #extract_info()
    #save_image()
    #output = extract_info()
    #annotations=convert_toyolo5(output)

    # Read images and annotations
    #images = [os.path.join('images', x) for x in os.listdir('dataset/images')]
    #annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

    #images.sort()
    #annotations.sort()

    # Split the dataset into train-valid-test splits 
    #train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    #val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    #Utility function to move images 


    # Move the splits into their folders
    #move_files_to_folder(train_images, 'images/train')
    #move_files_to_folder(val_images, 'images/val/')
    #move_files_to_folder(test_images, 'images/test/')
    #move_files_to_folder(train_annotations, 'annotations/train/')
    #move_files_to_folder(val_annotations, 'annotations/val/')
    #move_files_to_folder(test_annotations, 'annotations/test/')








    '''
    
    class_id_to_name_mapping = dict(zip(class_name.values(), class_name.keys()))



    # Get any random annotation file 
    annotation_file = random.choice(annotations)
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace("annotations", "dataset/images").replace("txt", "png")
    print("Annotation file path:", annotation_file)
    

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_bounding_box(image, annotation_list)
'''
   