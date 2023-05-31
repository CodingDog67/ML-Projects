import yaml
import torch
import shutil
import numpy as np
from glob import glob
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image

# first time restructure data
restructure = False

#structure data into data -> train -> images, mask, validation -> images, mask

# generic dataloader, dataset should be torch.dataset comform
def get_data_loader(data_set, batchsize, shuffle):
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size = batchsize,
        shuffle = shuffle
    )

    return data_loader


# create a datasets.Imagefolder object
class lgg_mri_dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir , transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):

        img_path = os.path.join(self.img_dir, self.images[idx]) 
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".tif", "_mask.tif"))

        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        mask[mask == 255.0] = 1.0 # easier for sigmoid 

        if self.transform:
            image = self.transform(image)
        
        return image, mask
    





def train_test_split(train_folder, test_folder, scanner_folders):
    pat_nums = len(scanner_folders)
    train_nums = int(np.floor(pat_nums * 0.7))
    
    train_folder.extend(scanner_folders[:train_nums])
    test_folder.extend(scanner_folders[train_nums:])

    return train_folder, test_folder

# moving files around to adapt to the pytorch dataset
def move_img(old_path, newPath):
    list_images= glob(old_path)
    for img in list_images:
        shutil.move(img, newPath) 


# run this only if you freshly downloaded the dataset, manually create the folder structur then occupy
# there are 4 scanner types CS, DU, FG, HT
if restructure:
    print(f"root dir: {os.getcwd()}")
    train_folder = []
    test_folder = []

    root_path = ".\\brain tumor segmentation LGG\\data\\kaggle_3m\\"

    scanners = ["*_CS_*", "*_DU_*", "*_FG_*", "*_HT_*"]

    # scanner and patientwise split
    for scanner in scanners:
        scan_fold = glob(root_path + scanner)
        train_folder, test_folder = train_test_split(train_folder, test_folder, scan_fold)

    for pat_folder in train_folder:
        # move masks first 
        move_img(pat_folder + "\\" + "*_mask.tif",
            '.\\brain tumor segmentation LGG\\data\\train_mask')
        # move remaining images 
        move_img(pat_folder + "\\" + "*.tif",
            '.\\brain tumor segmentation LGG\\data\\train_images')
        
    for pat_val_folder in test_folder:
        # move masks first 
        move_img(pat_val_folder + "\\" + "*_mask.tif",
            '.\\brain tumor segmentation LGG\\data\\val_mask')
        # move remaining images 
        move_img(pat_val_folder + "\\" + "*.tif",
            '.\\brain tumor segmentation LGG\\data\\val_images')


