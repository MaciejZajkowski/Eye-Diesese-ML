
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode



class EyeDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None,binary = False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.eye_frame = pd.read_csv(csv_file).drop(columns=['Unnamed: 0'])
        self.root_dir = root_dir
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + '/' + str(self.eye_frame.iloc[idx, 0])
        image = io.imread(img_name)
        
        label = self.eye_frame.iloc[idx, 1]
        if (self.binary and label > 0):
            label = 1
            
        sample = {'image': image, 'label': label}

        #print (f'image name:{img_name}, image shape{image.shape}')
        if self.transform:
            sample = self.transform(sample)
            
        #print (f'transformed image shape{sample["image"].shape}')
        
        return sample
    

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img, 'label': label}
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}
        
        
        
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    
    fig = plt.figure()
    fig.suptitle('Batch from dataloader')
    # fig, ax = plt.subplots(1,batch_size)
    fig.set_figheight(20)
    fig.set_figwidth(15)
    
    
    width = int(batch_size/2)
    height = int(batch_size/2)
    if (width + height) < batch_size:
        height = height +1
    for imnum in range(batch_size):
        ax=fig.add_subplot(width,height,imnum+1)
        ax.imshow(images_batch[imnum,:,:,:])
        ax.title.set_text(f'label #{sample_batched["label"][imnum]}')
        
    plt.show()