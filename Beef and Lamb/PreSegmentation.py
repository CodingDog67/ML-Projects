import glob
import Set_Workstation_Params
from segmentation_helper_functions import *
import SimpleITK as sitk

top_datapath, save_datapath = Set_Workstation_Params.setting_Workstation_Params(3)  # 1omen, 2home, 3carbon

ids = os.listdir(top_datapath)

#load ct images from an entire folder into a list for manipulation
for single_id in ids: # ids = single animals

    if single_id != "details.txt":
        sub_fold = top_datapath + single_id + "\\CT\\"
        sublist = os.listdir(sub_fold) #sub folder animal parts
    #bodypart loop
    for cur_dir in sublist:
        if cur_dir != "3010_20180718_101055.000":
            dir = sub_fold + cur_dir + "\\2\\"
            exp = os.listdir(dir) #toDO maybe only choose the highest current for now
            if "RequiredForDicomViewer.txt" in exp:
                exp.remove("RequiredForDicomViewer.txt")
            txt_info = [f for f in sorted(glob.glob(dir + exp[-1] + "\\" + "*.dcm"), key=numericalSort)]
            savepath_temp = save_datapath + single_id + "\\CT\\" + cur_dir + "\\2\\" + exp[-1] + "\\" + "Segmentation Results"
            if not os.path.exists(savepath_temp):
                os.makedirs(savepath_temp)

                single_scan_slices, dicom_sample = load_scan(txt_info)
                #single_scan_slices = convert_to_HU(single_scan_slices, dicom_sample)

                #checking the histogram over all scans
                plt.hist(single_scan_slices.flatten(), bins=50, color='c')
                plt.xlabel("Hu Units")
                plt.ylabel("frequency")
                plt.savefig(savepath_temp + '\\histogram.png', bbox_inches='tight')
                plt.close()

                # # #display a single image for checking reasons
                # norm = plt.Normalize(-1000, 1000)
                # plt.imshow(single_scan_slices[100, :, :], cmap="gray", interpolation='bicubic', norm=norm)
                # plt.show()

                # get the kmean clusters from a middle scan that has bone meat and fat
                #clusters, mean, std, maxi, mini = get_kmean_clusters(single_scan_slices[np.int(single_scan_slices.shape[0]/2), :, :])

                #display the entire stack /samples picks
                sample_stack_display(single_scan_slices, savepath_temp, display=False, labels=False)

                clusters = 0
                mean = 0
                std = 0
                maxi = 0
                mini = 0
                #segment entire scan
                label_map = segmentation_threshold(single_scan_slices, clusters, mean, std, maxi, mini, display=False)
                label_map = label_map.astype(np.uint8)
                sample_stack_display(label_map, savepath_temp, display=False, labels=True)
                label_map = np.flip(label_map, 0)

                spacing, origin = get_Spacing_origin(dir + exp[-1] + "\\")
                mhd_label = sitk.GetImageFromArray(label_map, isVector=False)
                mhd_label.SetSpacing(spacing)
                mhd_label.SetOrigin(origin)
                sitk.WriteImage(mhd_label, savepath_temp + "\\" + "Segmentedlabels.mhd", True)

            if os.path.exists(savepath_temp):
                label_map = sitk.GetArrayFromImage(sitk.ReadImage(savepath_temp + "\\Segmentedlabels.mhd"))
                vol_bones, vol_meat, vol_fat = get_volume(label_map)  #in cm³

                #weight in kg using standard average pig densities in g/cm³
                weight_bone = vol_bones * 1.433 /1000
                weight_meat = vol_meat * 1.117 /1000
                weight_fat = vol_fat * 0.997 / 1000

                mother_path = "E:\\DMRI_ABP\\CTdata\\"
                detail_file = open(savepath_temp + 'Segmentation_details.txt', 'w')


        plt.clf()
        plt.cla()
        plt.close()
        savepath_temp = ""
        testload = 0


