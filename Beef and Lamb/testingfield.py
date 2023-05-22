import SimpleITK as sitk
import glob
import pydicom
import numpy as np
from segmentation_helper_functions import *

save_path = "D:\\Data\\Beef Lamb\\results 5mA\\"

#default values
density_fat = 0.997
density_meat = 1.117
density_bone = 1.433
low_fat = -300
low_bone = 101
upper_fat = -10

data1new = pydicom.dcmread("C:\\Users\\Ari\\Desktop\\volume.dcm")
data2 = pydicom.dcmread("D:\\Data\\Beef Lamb\\Small_data_set\\3960\CT\\2934_20180716_151234.000\\2\\50\\1.2.392.200036.9116.2.5.1.16.1613454497.1531836341.104674.dcm", force=True)
data2.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

data2 = data2.pixel_array
data1new = data1new.pixel_array


#segment entire scan
label_map = segmentation_threshold(data1new, low_fat, upper_fat, low_bone, display=False)
label_map = label_map.astype(np.uint8)
sample_stack_display(data1new, save_path, display=False, labels=False)
sample_stack_display(label_map, save_path, display=False, labels=True)
label_map = np.flip(label_map, 0)

spacing, origin = get_Spacing_origin("D:\\Data\\Beef Lamb\\new 5mA data\\" + "\\")
mhd_label = sitk.GetImageFromArray(label_map, isVector=False)
mhd_label.SetSpacing(spacing)
mhd_label.SetOrigin(origin)
sitk.WriteImage(mhd_label, "D:\\Data\\Beef Lamb\\results 5mA" + "\\" + "Segmentedlabels.mhd", True)



pock = 0