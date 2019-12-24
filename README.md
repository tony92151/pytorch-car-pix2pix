
# Car edges-to-photo  pix2pix in PyTorch
This reop is forked in this [repe](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 
Our team's goal was replace dataset to car-dataset.

We use  [car-dataset from stanford](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 

First, we remove image's background by [Mask-Rcnn](https://github.com/facebookresearch/maskrcnn-benchmark). Mask-Rcnn help us gereate mask only for the biggest car in image and crop image by it's bounding box.
<img src='https://raw.githubusercontent.com/tony92151/pytorch-car-pix2pix/master/imgs/flow.jpg' width="600px"/>

# Data preprocessing

# Tarin
```python =
cd [path to pytorch-car-pix2pix]

python train.py --dataroot /home/u2546764/pytorch-car-pix2pix/datasets/car_data --name car_pix2pix --model pix2pix --direction BtoA --batch_size 2000 --gpu_ids 0,1,2,3,4,5,6,7 --num_threads 16
````

# Test
```python =
python test.py --dataroot /home/u2546764/pytorch-car-pix2pix/datasets/car_data --direction BtoA --model pix2pix --name car_pix2pix
```

# Facing problem
1. data preprocessing resizing
2. data i/o speed

# Hardware
```
32 core cpu
Telsa V100 *8
480G memory
```