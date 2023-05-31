# find data set here https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/code
# maybe denoise in the future but for now nothing to be done 
# MR bias field correction 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7790158/   for more information

import numpy as np

def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

def clip(scan):
    return np.clip(scan, -1, 2.5)
