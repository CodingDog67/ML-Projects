import yaml
import torch
import shutil
import glob
import os
from torchvision.datasets import ImageFolder
from torchvision.io import read_image


#structure data into data -> train -> tumor, non tumor, validation -> tumor and non tumor 

# moving files around to adapt to the pytorch dataset
def move_img(old_path, newPath):
    list_images= glob(old_path)
    for img in list_images:
        shutil.move(img, newPath) 

# generic dataloader, dataset should be torch.dataset comform
def get_data_loader(data_set, batchsize, shuffle):
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size = batchsize,
        shuffle = shuffle
    )

    return data_loader

# create a datasets.Imagefolder object
class data(torch.utils.data.Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.annotations = label_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(glob(self.img_dir))
    
    def __getitem__(self,idx):

        # ToDo  we get a list of patient folders that we need to unpack into one list and iterate through
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) ## change this
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label