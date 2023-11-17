## Object Detection and Semantic Segmentation on MNIST Double Digits RGB (MNISTDD-RGB)



### Overview

Engage in object detection and semantic segmentation tasks on the MNISTDD-RGB dataset. This dataset comprises RGB images, each containing two digits placed randomly.Objectives include identifying the digits in each image, determining their locations, and creating pixel-wise segmentation masks.

### Dataset

The MNISTDD-RGB dataset is split into three subsets: train, validation, and test, containing 55K, 5K, and 10K samples respectively. Each sample includes:
- **Image:** A 64×64×3 image vectorized to a 12288-dimensional vector.
- **Labels:** A 2-dimensional vector indicating the two digits in the image, always in ascending order.
- **Bounding Boxes:** A 2×4 matrix marking the locations of the two digits.
  --The first row contains the location of the first digit in labels and the second row contain the location of the second one. 
➢ Each row of the matrix has 4 numbers which represent [𝑦_𝑚𝑖𝑛, 𝑥_𝑚𝑖𝑛, 𝑦_𝑚𝑎𝑥, 𝑥_𝑚𝑎𝑥] in this exact order, where:
▪ 𝑦_𝑚𝑖n = row of the top left corner
▪ 𝑥_𝑚𝑖𝑛 = column of the top left corner
▪ 𝑦_𝑚𝑎𝑥 = row of the bottom right corner
▪ 𝑥_𝑚𝑎𝑥 = column of the bottom right corner
➢ It is always the case that 𝑥_𝑚𝑎𝑥 – 𝑥_𝑚𝑖𝑛 = 𝑦_𝑚𝑎𝑥 – 𝑦_𝑚𝑖𝑛 = 28. This means that each bounding box has a size of 
28 × 28 no matter how large or small the digit inside that box is.
- **Segmentation Mask:** A 64×64 image with pixel values ranging from 0 to 10, where 10 represents the background.

### Task

Provided with the train and validation sets for developing  methods.
- **pred_class:** Classification labels.
- **pred_bboxes:** Detection bounding boxes.
- **pred_seg:** Segmentation masks.
