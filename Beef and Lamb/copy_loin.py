' This script will only copy the loin part of each animal folder and copy it to a destination folder for less confusion ' \


import glob
from segmentation_helper_functions import *
from distutils.dir_util import copy_tree


# prompt for top data folder
top_datapath = input("Please input your custom Topmost Datapath for either Beef or Lamb\n")

if not top_datapath:
    while not top_datapath:
        top_datapath = input('please input a path: ')


#promt for data save path
save_datapath = input("Please input your path to your result folder (one for Beef and for Lamb must be created manually)\n")

if not save_datapath:
    while not save_datapath:
        save_datapath = input('please input a path: ')

# search all animal ids
ids = os.listdir(top_datapath)

#load ct images from an entire folder into a list for manipulation
for single_id in ids: # ids = single animals

    sub_fold = top_datapath + '\\' + single_id
    bodyPartList = os.listdir(sub_fold) #sub folder animal parts

    #bodypart loop
    for cur_dir in bodyPartList:
        if 'Loin' in cur_dir:

            fromDirectory = sub_fold + "\\" + cur_dir
            toDirectory = save_datapath + "\\" + single_id + "\\" + cur_dir

            if not os.path.exists(toDirectory):
                copy_tree(fromDirectory, toDirectory)