## Object Detection and Semantic Segmentation on MNIST Double Digits RGB (MNISTDD-RGB)

Welcome to Assignment 4 in CMPUT 328, where we explore the realms of object detection and semantic segmentation using the MNIST Double Digits RGB dataset.

### Overview

In this assignment, you will engage in object detection and semantic segmentation tasks on the MNISTDD-RGB dataset. This dataset comprises RGB images, each containing two digits placed randomly. Your objectives include identifying the digits in each image, determining their locations, and creating pixel-wise segmentation masks.

### Dataset

The MNISTDD-RGB dataset is split into three subsets: train, validation, and test, containing 55K, 5K, and 10K samples respectively. Each sample includes:
- **Image:** A 64×64×3 image vectorized to a 12288-dimensional vector.
- **Labels:** A 2-dimensional vector indicating the two digits in the image, always in ascending order.
- **Bounding Boxes:** A 2×4 matrix marking the locations of the two digits.
- **Segmentation Mask:** A 64×64 image with pixel values ranging from 0 to 10, where 10 represents the background.

### Task

Provided with the train and validation sets for developing  methods.
- **pred_class:** Classification labels.
- **pred_bboxes:** Detection bounding boxes.
- **pred_seg:** Segmentation masks.
