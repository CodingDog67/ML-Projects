# Copyright (c) 2012-2018 ImFusion GmbH, Munich, Germany. All rights reserved.
#
# This python script uses dcmdump to split DICOM files according to values of
# a DICOM tag into different subfolders
#
# Usage:
# splitdcm.py folder dicomtag
#
# Example:
# splitdcm.py /home/user/dicompics XRayTubeCurrent
# no need to enter path again, just put the command above without, edit path below in code
# make sur dcmdump is callable, add to environmental path, close and restart
#
# Author: Oliver Zettinig
# Date: 3 Aug 2018
#
# cd into python file then : python splitdcm.py XRayTubeCurrent

import subprocess
import os
import re
import sys
from os import listdir, walk
from os.path import isfile, join

#mother_path = "F:\\DMRI_ABP\\CTdata\\" # external harddrive
mother_path = "E:\\DMRI_ABP\\CTdata\\"
#mother_path = "D:\\PhD\\Project Management\\ETI Food\\Data\\Big Dataset\\CTdata\\"
list = os.listdir(mother_path)


for folder_name in list:

    sub_fold = mother_path + folder_name + "\\CT\\"
    sublist = os.listdir(sub_fold)

    for cur_dir in sublist:

# if len(sys.argv) < 3:
#     print('Usage: splitdcm.py folder dicomtag')
#     sys.exit(1)

        if cur_dir != "3010_20180718_101055.000":
            dir = sub_fold + cur_dir + "\\2\\"
            tag = "XRayTubeCurrent"
            print(dir)
            #dir = sys.argv[1]
            #tag = sys.argv[2]


        for filename in os.listdir(dir):
            if filename.endswith('.dcm'):
                fp = os.path.join(dir, filename)
                res = subprocess.check_output(['dcmdump', fp]).decode('UTF-8').strip()
                pos = res.find(tag)
                m = re.search('\[.*\].*' + tag, res)
                res = m.group(0)
                subdirname = res[res.find('[')+1:res.find(']')]
                fullsubdir = os.path.join(dir, subdirname)
                if not os.path.exists(fullsubdir):
                    os.mkdir(fullsubdir)
                os.rename(fp, os.path.join(fullsubdir, filename))

                filename2 = filename.replace('.dcm', '.txt')
                fp2 = os.path.join(dir, filename2)
                if os.path.exists(fp2):
                    os.rename(fp2, os.path.join(fullsubdir, filename2))

