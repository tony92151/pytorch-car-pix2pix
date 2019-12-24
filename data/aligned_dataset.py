import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import time
import datetime


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        

        img = Image.open(self.AB_paths[0]).convert('RGB')
        w, h = img.size
        w2 = int(w / 2)
        self.A = []
        self.B = []
        print("================================================")
        print("Load images to memory")
        t = time.time()
        for i in range(len(self.AB_paths)):
            image = Image.open(self.AB_paths[i]).convert('RGB')
            A = image.crop((0, 0, w2, h))
            B = image.crop((w2, 0, w, h))
            transform_params = get_params(self.opt, A.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            A = A_transform(A)
            B = B_transform(B)
            self.A.append(A)
            self.B.append(B)
        print("Done")
        print("Take",datetime.timedelta(seconds = int(time.time()-t)),"seconds")
        print("================================================")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        #AB = Image.open(AB_path).convert('RGB')
        #AB = Image.fromarray(cv2.cvtColor(self.AB[index],cv2.COLOR_BGR2RGB)
        #AB = self.AB[index]
        # split AB image into A and B
        #w, h = AB.size
        #w2 = int(w / 2)
        #A = AB.crop((0, 0, w2, h))
        #B = AB.crop((w2, 0, w, h))
        #A = self.A[index]
        #B = self.B[index]
                          
        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        #A = A_transform(A)
        #B = B_transform(B)
        A = self.A[index]
        B = self.B[index]
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
