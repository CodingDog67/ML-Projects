# find the slides which are annotated
# write all 110 into one mat file
# write a corresponding label file
# save as .png

# [Cls 1:] Region above the retina (RaR);
# [Cls 2:] ILM: Inner limiting membrane;
# [Cls 3:] NFL-IPL: Nerve fiber ending to Inner plexiform layer;
# [Cls 4:] INL: Inner Nuclear layer;
# [Cls 5:] OPL: Outer plexiform layer;
# [Cls 6:] ONL-ISM: Outer Nuclear layer to Inner segment myeloid;
# [Cls 7:] ISE: Inner segment ellipsoid;
# [Cls 8:] OS-RPE: Outer segment to Retinal pigment epithelium;
# [Cls 9:] Region below RPE (RbR)
# [Cls 10:] Fluid region

import scipy.io as sio
from glob import glob
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.misc
import imageio
import cv2
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='2015_BOE_Chiu',
                        help='dataset name')
    args = parser.parse_args()

    return args

def interpolate(annotations, line , nan_locs):
    start = nan_locs[0]
    dis_num = 0
    for i in range(1, len(nan_locs)):
        distance = nan_locs[i] - nan_locs[i-1]
        dis_num += 1
        if distance == 1 and i!= (len(nan_locs)-1):
          continue

        if i == (len(nan_locs)-1):
            if distance == 1:
                end = nan_locs[i]
                dis_num += 1

            else:
                end = nan_locs[i-1]

                # interpolate and replace nans
                arr = np.linspace(annotations[line, start - 1], annotations[line, end + 1], num=dis_num, endpoint=False)
                arr = arr.round()
                annotations[line, start:end + 1] = arr

                # do the same for last value of array
                start = nan_locs[i]
                end = nan_locs[i]
                dis_num = 1

                arr = np.linspace(annotations[line, start - 1], annotations[line, end + 1], num=dis_num, endpoint=False)
                arr = arr.round()
                annotations[line, start:end + 1] = arr

                continue


        else:
            end = nan_locs[i-1]

        #interpolate and replace nans
        arr = np.linspace(annotations[line, start-1], annotations[line, end+1], num=dis_num, endpoint=False)
        arr=arr.round()
        annotations[line, start:end+1] = arr

        #update to a new start and reset dis_num
        start = nan_locs[i]
        dis_num = 0

    return annotations

#Load MATLAB file
args = parse_args()
file_paths = glob('input\\' + args.dataset + '\\*')
fix_nan = True
with_fluid = True
annotator = 1
visualize = False
crop_top = 40
crop_bot = 50

if annotator == 1:
    layer_annot = 'manualLayers1'
    fluid_annot = 'manualFluid1'

else:
    layer_annot = 'manualLayers2'
    fluid_annot = 'manualFluid2'

for file_name in file_paths:
    file = sio.loadmat(file_name)
    pat_num = file_name.replace('input\\' + args.dataset + '\\', '')
    pat_num = pat_num.replace('.mat', '')
    #get annotated slides location
    annot_slices_loc = np.unique(np.where(~np.isnan(file[layer_annot]))[2])

    #save individual image and entire patient
    #name: pat num + slice num
    for slice in annot_slices_loc:
        image = file['images'][:, :, slice]
        img_name = pat_num + '_' + str(slice)
        annotation = file[layer_annot][:, :, slice]
        fluid = file[fluid_annot][:, :, slice]


        #check if the annotations start at the same loc and save that down
        first = np.unique(np.where(~np.isnan(file[layer_annot][0, :, slice])))[0]
        last = np.unique(np.where(~np.isnan(file[layer_annot][0, :, slice])))[-1]

        # check if all annotation lines have the same length
        if img_name == 'Subject_04_24' and slice == 24:
            continue
        for i in range(1, annotation.shape[0]):
            first_temp = np.unique(np.where(~np.isnan(file[layer_annot][i, :, slice])))[0]
            last_temp = np.unique(np.where(~np.isnan(file[layer_annot][i, :, slice])))[-1]
            if first != first or last != last_temp:
                print("non consistent annotation start/end")
                quit()

        #crop image to fit annotations
        image = image[:, first:last]
        annotation = annotation[:, first:last]
        fluid = fluid[:, first:last]


        fluid[fluid > 0] = 10
        #interpolate between missing annotations
        if fix_nan == True:
            nan_vals = (np.where(np.isnan(annotation)))
            which_lines = np.unique(nan_vals[0], return_counts=True)
            count = 0
            for i in range(len(which_lines[0])):
                annotation = interpolate(annotation, which_lines[0][i], nan_vals[1][count:(count + which_lines[1][i])])
                count += which_lines[1][i]

        annotation = annotation.astype(int)
        final_annot = np.zeros(image.shape)

        for l in range((last-first)):
            #final_annot[:annot_loc[0, l], l] = 10
            final_annot[(annotation[7, l]):, l] = 1
            final_annot[(annotation[6, l]):annotation[7, l], l] = 2
            final_annot[(annotation[5, l]):annotation[6, l], l] = 3
            final_annot[(annotation[4, l]):annotation[5, l], l] = 4
            final_annot[(annotation[3, l]):annotation[4, l], l] = 5
            final_annot[(annotation[2, l]):annotation[3, l], l] = 6
            final_annot[(annotation[1, l]):annotation[2, l], l] = 7
            final_annot[(annotation[0, l]):annotation[1, l], l] = 8
            final_annot[:annotation[0, l], l] = 9

        # crop a bit off top and bot
        image = image[crop_top:(image.shape[0] - crop_bot), :]
        final_annot = final_annot[crop_top:(final_annot.shape[0] - crop_bot), :]
        fluid = fluid[crop_top:(fluid.shape[0] - crop_bot), :]

        if with_fluid:
            final_annot = final_annot + fluid
            final_annot[final_annot > 10] = 10
            final_annot = final_annot.astype(np.uint8)
            #final_annot = cv2.normalize(final_annot, 0, 255, norm_type=cv2.NORM_MINMAX)

            if not os.path.exists('input\\%s' % args.dataset + "_images"):
                os.makedirs('input\\%s' % args.dataset + "_images")
                os.makedirs('input\\%s' % args.dataset + "_masks1")

            annot_name = img_name + "_label_" + str(annotator)

            imageio.imwrite('input\\' + args.dataset + "_images\\" + img_name + ".png", image)
            imageio.imwrite('input\\' + args.dataset + "_masks1\\" + annot_name + ".png", final_annot)

        else:
            final_annot = final_annot.astype(np.uint8)
            #final_annot = cv2.normalize(final_annot, 0, 255, norm_type=cv2.NORM_MINMAX)

            if not os.path.exists('input\\%s' % args.dataset + "_images\\separated"):
                os.makedirs('input\\%s' % args.dataset + "_masks1\\separated")
                os.makedirs('input\\%s' % args.dataset + "_images\\separated")

            fluid_name = img_name + "_fluidLabel_" + str(annotator)
            annot_name = img_name + "_label_" + str(annotator)

            imageio.imwrite('input\\' + args.dataset + "_masks1\\separated\\" + fluid_name + ".png", fluid)

            imageio.imwrite('input\\' + args.dataset + "_masks1\\separated\\" + annot_name + ".png", final_annot)
            imageio.imwrite('input\\' + args.dataset + "_images\\separated\\" + img_name + ".png", image)

        if visualize:
            plt.imshow(image)
            plt.show()
            plt.imshow(final_annot)
            plt.show()
            plt.imshow(image)
            plt.imshow(final_annot, alpha=0.3)
            plt.show()
            plt.close('all')

        gc.collect()


