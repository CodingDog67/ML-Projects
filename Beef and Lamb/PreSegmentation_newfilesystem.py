import glob
from segmentation_helper_functions import *
import SimpleITK as sitk
from pandas import DataFrame
from scipy.sparse import csr_matrix

#fasten computation of thresholding
def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

top_datapath = input("Please input your custom Topmost Datapath for either Beef or Lamb\n")

if not top_datapath:
    while not top_datapath:
        top_datapath = input('please input a path: ')

save_datapath = input("Please input your path to your result folder (one for Beef and for Lamb must be created manually)\n")

if not save_datapath:
    while not save_datapath:
        save_datapath = input('please input a path: ')


ids = os.listdir(top_datapath)

#default values

density_fat = 0.99    # 0.9             # 0.997
density_meat = 1.12   # 1.1             # 1.117
density_bone = 1.43  # 1.41  for lamb  # 1.433 pig
low_fat = -235 #default -300
upper_fat = 2 #default -10
low_bone = 160  # default 101
low_meat = upper_fat + 1
upper_meat = low_bone - 1

writelatex = False
#empty list for excel list

ids_list = []
actual_weight = []
est_weight = []
bone_weight = []
lean_weight = []
fat_weight = []
bone_percent = []
lean_percent = []
fat_percent = []
current = []



#load ct images from an entire folder into a list for manipulation
for single_id in ids: # ids = single animals

    sub_fold = top_datapath + '\\' + single_id
    bodyPartList = os.listdir(sub_fold) #sub folder animal parts

    #bodypart loop
    for cur_dir in bodyPartList:

        scandir = sub_fold + "\\" + cur_dir
        txt_info = [f for f in sorted(glob.glob(scandir + "\\" + "*.dcm"), key=numericalSort)]
        savepath_temp = save_datapath + "\\" + single_id + "\\" + cur_dir + "\\" + "Segmentation Results"

        if not os.path.exists(savepath_temp):
            os.makedirs(savepath_temp)

            single_scan_slices, dicom_sample = load_scan(txt_info)
            #single_scan_slices = convert_to_HU(single_scan_slices, dicom_sample)

            #display the entire stack /samples picks
            sample_stack_display(single_scan_slices, savepath_temp, display=False, labels=False)

            #segment entire scan
            label_map = segmentation_threshold(single_scan_slices, low_fat, upper_fat, low_bone, display=False)
            label_map = label_map.astype(np.uint8)
            sample_stack_display(label_map, savepath_temp, display=False, labels=True)
            label_map = np.flip(label_map, 0)

            spacing, origin = get_Spacing_origin(scandir + "\\")
            mhd_label = sitk.GetImageFromArray(label_map, isVector=False)
            mhd_label.SetSpacing(spacing)
            mhd_label.SetOrigin(origin)
            sitk.WriteImage(mhd_label, savepath_temp + "\\" + "Segmentedlabels.mhd", True)

        if os.path.exists(savepath_temp):
            label_map = sitk.GetArrayFromImage(sitk.ReadImage(savepath_temp + "\\Segmentedlabels.mhd"))
            vol_bones, vol_meat, vol_fat, percent_bone, percent_lean, percent_fat = get_volume(label_map)  #in cm³

            #weight in kg using standard average pig densities in g/cm³
            weight_bone = vol_bones * density_bone /1000
            weight_meat = vol_meat * density_meat /1000
            weight_fat = vol_fat * density_fat / 1000


            overall_weight = weight_fat + weight_meat + weight_bone

            #detail_file = open(savepath_temp + "\\" + 'Segmentation_details.txt', 'w') #folderwise save

            cur = "sth"
            if "lamb" in save_datapath:
                if "mA30" in cur_dir:
                    cur = "30mA"
                else:
                    cur = "60mA"
            if "beef" in save_datapath:
                cur = cur_dir

            #latex info file
            if writelatex:
                latex_file = open(save_datapath + "\\" + 'latex.txt', 'a')
                latex_file.write("\n" + single_id + " & " + "actualWeight" + " & " + str(round(overall_weight, 2)) + " & "
                                  + str(round(weight_bone, 2)) + " / " + str(percent_bone) + " & "
                                  + str(round(weight_meat, 2)) + " / " + str(percent_lean) + " & "
                                  + str(round(weight_fat, 2)) + " / " + str(percent_fat) + " & " + cur + "   \\\\ \\hline")

                latex_file.close()

            ids_list.append(single_id)
            actual_weight.append(0)
            est_weight.append((round(overall_weight, 2)))
            bone_weight.append((round(weight_bone, 2)))
            lean_weight.append((round(weight_meat, 2)))
            fat_weight.append((round(weight_fat, 2)))
            bone_percent.append(percent_bone)
            lean_percent.append(percent_lean)
            fat_percent.append(percent_fat)
            current.append(cur)


            # with open(save_datapath+ "\\" + 'Segmentation_details.txt', 'a') as detail_file:
            #     detail_file.write("\nAnimal ID:" + single_id + "\n")
            #     detail_file.write(
            #         "weights in Kilogram" + "\n" + "Bone: " + str(round(weight_bone, 2)) + "\n" + "Meat: " + str(
            #             round(weight_meat, 2)) + "\n" + "Fat: " + str(round(weight_fat, 2)) + "\n" + "overall Weight: " + str(round(overall_weight, 2)) + "\n")
            #
            #     detail_file.write("percentage of bone, meat and fat compared to overall volume " + str(percent_bone) + " " + str(percent_lean) + " " + str(percent_fat) + "\n")
            #
            #     detail_file.write("Volume of bone, meat and fat in g/cm³: " + str(round(vol_bones, 2)) + " " + str(
            #         round(vol_meat, 2)) + " " + str(round(vol_fat, 2)) + "\n")
            #
            #     if single_id == "4238":
            #         detail_file.write("Based on following average densities, bone: " + str(density_bone) + ", meat: " + str(density_meat) + ", fat: " + str(density_fat) + "\n")
            #
            #         detail_file.write("Based on following thresholds, fat: " + str(low_fat) + " to " + str(upper_fat) + "\n"
            #                         + "meat: " + str(upper_fat + 1) + " to " + str(low_bone - 1) + "\n"
            #                         + "bone everything over: " + str(low_bone))


# excel sheet information
df = DataFrame({'Animal ID': ids_list, 'Actual Weight': actual_weight, 'Estimated Weight': est_weight,
                'Bone Kg': bone_weight, 'Percent bone': bone_percent,
                'Lean Kg': lean_weight, 'Percent lean': lean_percent,
                'Fat Kg': fat_weight, 'Percent fat': fat_percent,
               'Current': current})

df.to_excel(save_datapath + "\\" + 'lamb_details.xlsx', sheet_name='sheet1', index=True)



