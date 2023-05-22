import glob
from segmentation_helper_functions import *
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

'Script to delete all mislabeled air pixels' \
'loaded labels need to be flipped and flipped again before saving itk number is backwards'

#testing
def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


#top_datapath = input("Please input your custom Topmost Datapath for Labels")

top_datapath = 'C:\\Users\\Ari\\Desktop\\test_label' # comment out later

if not top_datapath:
    while not top_datapath:
        top_datapath = input('please input a path: ')


ids = os.listdir(top_datapath)
scan_ids = next(os.walk(top_datapath))[
    1]  # sub folder animal parts [1] returns subfolders, 0 motherpath, 2 files in folder

#default values

density_fat = 0.99    # 0.9             # 0.997
density_meat = 1.12   # 1.1             # 1.117
density_bone = 1.43  # 1.41  for lamb  # 1.433 pig
low_fat = -350 #default -300
upper_fat = 2 #default -10
low_bone = 160  # default 101
low_meat = upper_fat + 1
upper_meat = low_bone - 1

writelatex = False
#empty list for excel list


#load ct images from an entire folder into a list for manipulation
for single_id in scan_ids: # ids = single animals

    #todo write code to read out animal number, navigate to data folder with same number load img,
    # find all air voxels

    sub_fold = os.listdir(top_datapath + '\\' + single_id)

    for cur in sub_fold:
        if "mA50" in cur:  #chose the currency here
            scandir = top_datapath + '\\' + single_id + '\\' + cur

            txt_info = [f for f in sorted(glob.glob(scandir + "\\" + "*.dcm"), key=numericalSort)]
            single_scan_slices, dicom_sample = load_scan(txt_info)


            reader = sitk.ImageFileReader()
            reader.SetImageIO("MetaImageIO")
            reader.SetFileName(top_datapath + "\\" + "label_" + single_id + "_questions" + ".mha")
            mhd_label = sitk.GetArrayFromImage(reader.Execute())
            mhd_label = np.flip(mhd_label, 0)
            #test
            norm = plt.Normalize(-1000, 1000)
            plt.imshow(single_scan_slices[80, :, :], cmap='gray', interpolation='bicubic')
            plt.show()
            plt.imshow(mhd_label[80, :, :])
            plt.show()

            # t = time.time()
            # air_pixels = np.where(single_scan_slices < low_fat, 1.0, 0.0)  # air
            # elapsed = time.time() - t
            # print(elapsed)

            t = time.time()
            air_pixels2 = single_scan_slices <low_fat
            elapsed = time.time() - t
            print(elapsed)

            mhd_label[air_pixels2 == True] = 0  # air

            mhd_label = mhd_label.astype(np.uint8)
            mhd_label = np.flip(mhd_label, 0)

            spacing, origin = get_Spacing_origin(scandir + "\\")
            mhd_label = sitk.GetImageFromArray(mhd_label, isVector=False)
            mhd_label.SetSpacing(spacing)
            mhd_label.SetOrigin(origin)
            sitk.WriteImage(mhd_label, top_datapath + "\\" + single_id + "_corrected.mhd", True)






