
#1 = laptop OMEN
#2 = homeworkstation
#3 = laptop carbon

saved_data_path = ""

def set_data_load_path(path_num):
    switcher = {
        1: "D:\\PhD\\Project Management\\ETI Food\Data\\Big Dataset\\CTdata\\",
        2: "I:\\DMRI_ABP\\CTdata\\",
        #3: "F:\\DMRI_ABP\\CTdata\\" #external hdd
        3: "D:\\Data\\Beef Lamb\\Small_data_set\\"  #local
    }

    return switcher.get(path_num, 'nothing')


def set_data_save_path(path_num):
    switcher = {
        1: "D:\\PhD\\Project Management\\ETI Food\Data\\Big Dataset\\Presegmentation_data\\",
        2: "something",
        #3: "F:\\DMRI_ABP\\Segmentation Labels\\"
        3: "D:\\Data\\Beef Lamb\\small set pre segmentations\\" #local
    }

    return switcher.get(path_num, 'nothing')


def setting_Workstation_Params(path_num):

    load_Path = set_data_load_path(path_num)
    saved_data_path = set_data_save_path(path_num)

    return load_Path, saved_data_path

