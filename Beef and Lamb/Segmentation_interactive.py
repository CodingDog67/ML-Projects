import os
import glob
from segmentation_helper_functions import *
import SimpleITK as sitk


#to create an exucuteable use the following
# open console and use " pyinstaller -F yourprogram.py" -F to have one executable

#default values
density_fat = 0.997
density_meat = 1.117
density_bone = 1.433

loop_value = True
backup_datapath = ''
backup_savepath = ''

# USER INPUT
while loop_value:
    continue_check = input("\nIf you want to continue leave this blank, else enter anything \n")
    if continue_check:
        loop_value = False
        exit()
        break

    scan_datapath = input('\nEnter the Scan folder path, leave it blank if you want to keep the old path: \n')


    if scan_datapath:
        backup_datapath = scan_datapath  # keep old datapath
    elif not scan_datapath and backup_datapath:
        scan_datapath = backup_datapath
    else:
        while not scan_datapath and not backup_datapath:
            scan_datapath = input('please input a path: ')

    save_datapath = input('\nEnter the save folder path, leave it blank if you want to keep the old path: \n')
    if save_datapath:
        backup_savepath = save_datapath
    elif not save_datapath and backup_savepath:
        save_datapath = backup_savepath
    else:
        while not save_datapath and not backup_savepath:
            save_datapath = input('please input a path: ')


    savepath_temp = save_datapath + "\\" + "Segmentation Results"   #toDo remember to add the same path style as input path

    print("\nDefault thresholds are: fat -235 to -2, meat -1 to 159, bone > 159, enter nothing for either of following if you want to keep them")
    low_fat = input('Enter lower fat threshold: ')
    upper_fat = input('Enter upper fat threshold: ')
    low_bone = input('Enter lower bone threshold: ')


    # default thresholds
    if not low_fat or not low_bone or not upper_fat:
        low_fat = -235
        low_bone = 160
        upper_fat = -2
        resultname = str(low_fat) + "_" + str(upper_fat) + "_" + str(low_bone) + "_"

    else:
        resultname = low_fat + "_" + upper_fat + "_" + low_bone + "_"
        low_fat = int(float(low_fat))
        upper_fat = int(float(upper_fat))
        low_bone = int(float(low_bone))



    print("\nDefault density values are - fat: 0.997,meat: 1.117, bone: 1.433, enter nothing if you want to keep them")
    density_fat = input("Enter the density for fat: ")
    density_meat = input("Enter the density for meat: ")
    density_bone = input("Enter the density for bone: ")

    #default densities based on pig meat study (dennis)
    if not density_fat or not density_meat or not density_bone == 0:
        density_fat = 0.997
        density_meat = 1.117
        density_bone = 1.433

    else:
        [density_fat, density_meat, density_bone] = ["{0:.3f}".format(round(float(x), 3)) for x in [density_fat, density_meat, density_bone]]


    low_meat = upper_fat + 1
    upper_meat = low_bone - 1

    # SEGMENTATION PART
    #load ct images from an entire folder into a list for manipulation
    txt_info = [f for f in sorted(glob.glob(scan_datapath + "\\" + "*.dcm"), key=numericalSort)]

    single_scan_slices, dicom_sample = load_scan(txt_info)
    patientID = dicom_sample.PatientID

    savepath_temp = save_datapath + "\\" + patientID + "_Segmentation Results"  # toDo remember to add the same path style as input path

    if not os.path.exists(savepath_temp):
        os.makedirs(savepath_temp)

    # display the entire stack /samples picks
    sample_stack_display(single_scan_slices, savepath_temp, display=False, labels=False)

    # segment entire scan

    label_map = segmentation_threshold(single_scan_slices, low_fat, upper_fat, low_bone, display=False)
    label_map = label_map.astype(np.uint8)
    sample_stack_display(label_map, savepath_temp, display=False, labels=True)
    label_map = np.flip(label_map, 0)

    spacing, origin = get_Spacing_origin(scan_datapath)
    mhd_label = sitk.GetImageFromArray(label_map, isVector=False)
    mhd_label.SetSpacing(spacing)
    mhd_label.SetOrigin(origin)
    sitk.WriteImage(mhd_label, savepath_temp + "\\" + resultname + "Segmentedlabels.mhd", True)

    if os.path.exists(savepath_temp):
        label_map = sitk.GetArrayFromImage(sitk.ReadImage(savepath_temp + "\\" + resultname + "Segmentedlabels.mhd"))
        vol_bones, vol_meat, vol_fat, percent_bone, percent_lean, percent_fat = get_volume(label_map)  # in cm³

        # weight in kg using standard average pig densities in g/cm³
        weight_bone = vol_bones * density_bone / 1000
        weight_meat = vol_meat * density_meat / 1000
        weight_fat = vol_fat * density_fat / 1000

        overall_weight = weight_fat + weight_meat + weight_bone

        detail_file = open(save_datapath + "\\" + 'Segmentation_details.txt', 'w')
        detail_file.write(
            "weights in Kilogram" + "\n" + "Bone: " + str(round(weight_bone, 2)) + "\n" + "Meat: " + str(
                round(weight_meat, 2)) + "\n" + "Fat: " + str(round(weight_fat, 2)) + "\n" + "overall Weight: " + str(
                round(overall_weight, 2)) + "\n")

        detail_file.write(
            "percentage of bone, meat and fat compared to overall volume " + str(percent_bone) + " " + str(
                percent_lean) + " " + str(percent_fat) + "\n")

        detail_file.write("Volume of bone, meat and fat in g/cm³: " + str(round(vol_bones, 2)) + " " + str(
            round(vol_meat, 2)) + " " + str(round(vol_fat, 2)) + "\n")

    plt.clf()
    plt.cla()
    plt.close()


