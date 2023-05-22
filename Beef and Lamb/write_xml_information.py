# Copyright (c) TUM  Munich, Germany. All rights reserved.
#
# This python script read in individual scanning parameters and combines them in a summary info file
#
# Usage:
# adjust hardcoded mother path and run
#
# Author: Arianne Tran
# Date: 8 Aug 2018
#


import subprocess
import os
import re
import sys
import glob
from os import listdir, walk
from os.path import isfile, join
from openpyxl import Workbook

mother_path = "E:\\DMRI_ABP\\CTdata\\"
detail_file = open(mother_path + 'details.txt', 'w')

ids = os.listdir(mother_path)


for single_id in ids:

    if single_id != "details.txt":
        sub_fold = mother_path + single_id + "\\CT\\"
        sublist = os.listdir(sub_fold)

    for cur_dir in sublist:

# if len(sys.argv) < 3:
#     print('Usage: splitdcm.py folder dicomtag')
#     sys.exit(1)

        if cur_dir != "3010_20180718_101055.000":
            dir = sub_fold + cur_dir + "\\2\\"
            exp = os.listdir(dir)
            current_str = ""
            for i in range(len(exp)-1):
                current_str += exp[i] + " "
            txt_info = [f for f in glob.glob(dir + exp[0] + "\\" + "*.txt")][0]
            print(dir)
            read_file = open(txt_info, "r+")
            animal = read_file.readlines()[31]
            bodypart = "entire body\n"

            if 'ACQ Protocol Name=ABP -' in animal:
                animal = re.sub('^ACQ Protocol Name=ABP -', '', animal)
                animal = animal[:-1]

            if "2" in animal or "1" in animal:
                read_file.seek(0)  # reset the file pointer to beginning of file
                bodypart = read_file.readlines()[20]
                if 'PAT Patient Name=' in bodypart:
                    bodypart = re.sub('^PAT Patient Name=', '', bodypart)
                animal = "Beef"

            detail_file.write("ID: " + single_id + " Currents: " + current_str + " Animal: " + animal + " Bodypart: " + bodypart + "\n")
            #for filename in os.listdir(dir):

detail_file.close()