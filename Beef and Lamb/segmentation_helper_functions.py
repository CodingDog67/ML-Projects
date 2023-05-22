import pydicom
import numpy as np
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import measure
from sklearn.cluster import KMeans
from plotly.offline import iplot
from plotly.tools import FigureFactory as FF
import os

hiddenimports = []

#sorting loaded files by increasing numerical value instead of alphabethical
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#load single scan
def load_scan(path):
    dicom_data = [(pydicom.dcmread(s, force=True)) for s in path]

    dicom_data = sorted(dicom_data, key=lambda x: x.InstanceNumber)
    for single_slide in dicom_data:
        single_slide.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    slices = np.stack([s.pixel_array for s in dicom_data]) #list comprehension from now on
    sample_data = dicom_data[0]

    return slices, sample_data

def convert_to_HU(scans, dicom_info):

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    # 2048 outside picture range
    image = scans.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2048] = image[0, np.int(image.shape[1]/2), 0]

    # Convert to Hounsfield units (HU)
    intercept = dicom_info.RescaleIntercept
    slope = dicom_info.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return image

#display stack of imgs over scan
def sample_stack_display(stack, savepath, display = True, rows=6, cols=6, start_with=10, save=True, labels = False):
    fig, ax = plt.subplots(rows, cols, figsize=[13, 13])
    show_every = np.int((np.floor(stack.shape[0] - start_with) / (rows * cols)))
    for i in range(rows * cols):
        norm = plt.Normalize(-1000, 1000)
        ind = start_with + i*show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        if labels == False:
            ax[int(i / rows), int(i % rows)].imshow(stack[ind, :, :], cmap='gray', interpolation='bicubic', norm=norm)
        else:
            ax[int(i / rows), int(i % rows)].imshow(stack[ind, :, :])

        ax[int(i / rows), int(i % rows)].axis('off')

    if save:
        if labels == False:
            if os.path.isfile(savepath + '\\slices.png'):
                os.remove(savepath + '\\slices.png')
            plt.savefig(savepath + '\\slices.png', bbox_inches='tight')
        else:
            if os.path.isfile(savepath + '\\label_slices.png'):
                os.remove(savepath + '\\label_slices.png')
            plt.savefig(savepath + '\\label_slices.png', bbox_inches='tight')

    if display:
        plt.show()


#resample the data so one pix represents 1mm x 1mm + 1mm, compare btw diff ct scans display in 3D isometric form
def resample(img, img_stack, new_spacing=[1, 1, 1]):
    # get cur spacing
    spacing = map(float, (img_stack[0].SliceThickness+img_stack[0].PixelSpacing))
    spacing = (np.array(list(spacing)))

    resize_factor = spacing/new_spacing
    new_real_shape = np.round(img.shape*resize_factor)
    real_resize_fac = new_real_shape / img.shape
    new_spacing = spacing / real_resize_fac  # just to check should be 1, 1, 1

    image = scipy.ndimage.interpolation.zoom(img, real_resize_fac)

    return image, new_spacing


def make_mesh(image, threshold=-300, step_size=1):
    print
    "Transposing surface"
    p = image.transpose(2, 1, 0)

    print
    "Calculating surface"
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print
    "Drawing"

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)


def plt_3d(verts, faces):
    print
    "Drawing"
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()

def normalize_img(img, mean, std, max, min):
    norm = plt.Normalize(-1000, 1000)
    cmap = plt.cm.gray
    img = cmap(norm(img))
    img = img[:,:,1]
    img.flatten()
    # plt.imshow(single_scan_slices[100, :, :], cmap="gray", interpolation='bicubic', norm=norm)

    row_size = img.shape[0]
    col_size = img.shape[1]

    # normalize pix value by mean and std
    #img = img - mean
    #img = img / std

    # # Find the average pixel value near the lungs
    # # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    if max == 0 and min == 0:
        max = np.max(img)
        min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum

    #img[img == max] = mean
    #img[img == min] = mean

    return img, middle, max, min


def get_kmean_clusters(img):

    mean = np.mean(img)
    std = np.std(img)
    img, middle, maxi, mini = normalize_img(img, mean, std, 0, 0)

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=4).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold_1 = np.mean(centers[0:2])  # meat
    threshold_2 = np.mean([np.mean([np.mean(centers[1:3]), centers[2]]),
                           centers[2]])  # one third not the middle, more sensitive towards fat
    threshold_3 = np.mean(centers[2:4])  #


    return [threshold_1, threshold_2, threshold_3], mean, std, maxi, mini

#returns spacing/org for writing the correct mhd label file, more clever solution might exist
def get_Spacing_origin(filepath):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    spacing = image3D.GetSpacing()
    origin = image3D.GetOrigin()

    return spacing, origin

def segmentation_threshold(img_stack, low_fat, upper_fat, low_bone, display=False):

    low_meat = upper_fat + 1
    upper_meat = low_bone - 1

    slice_num = img_stack.shape[0]

    label_stack = np.empty(((img_stack.shape)))
    for i in range(slice_num):

        img = img_stack[i, :, :]

        #trial to hardcode the thresholds
        #thresh_img1 = np.where(img < low_fat, 1.0, 0.0)  # air
        thresh_img1 = img < low_fat  # air
        thresh_img2 = np.where(np.logical_and(img >= low_fat, img <= upper_fat), 1.0, 0.0)  # fat -100 - 50
        thresh_img3 = np.where(np.logical_and(img >= low_meat, img <= upper_meat), 1.0, 0.0)  # meat 50 100
        #thresh_img4 = np.where(img > low_bone, 1.0, 0.0)
        thresh_img4 = img > low_bone  # bone 200 +

        labels = np.zeros((img.shape[0], img.shape[1]))
        labels[thresh_img1 == True] = 0  # air
        labels[thresh_img2 == 1] = 1  # fat
        labels[thresh_img3 == 1] = 2  # meat
        labels[thresh_img4 == True] = 3  # bone
        labels = labels.astype(int)
        label_vals, count  = np.unique(labels, return_counts=True)

        label_stack[i, :, :] = labels

        if display:
            fig, ax = plt.subplots(3, 2, figsize=[12, 12])
            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(img, cmap='gray')
            ax[0, 0].axis('off')
            ax[0, 1].set_title("Color Labels")
            ax[0, 1].imshow(labels)
            ax[0, 1].axis('off')
            ax[1, 0].set_title("Air")
            ax[1, 0].imshow(thresh_img1, cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 1].set_title("Bone")
            ax[1, 1].imshow(thresh_img4, cmap='gray')
            ax[1, 1].axis('off')
            ax[2, 0].set_title("Fat")
            ax[2, 0].imshow(thresh_img2, cmap='gray')
            ax[2, 0].axis('off')
            ax[2, 1].set_title("Meat")
            ax[2, 1].imshow(thresh_img3, cmap='gray')
            ax[2, 1].axis('off')

            plt.show()

    label_stack = label_stack.astype(np.int32)
    return label_stack

#volume in cm³
def get_volume(label_map):
    label_vals, [air, num_fat, num_meat, num_bones] = np.unique(label_map, return_counts=True)

    vol_bones = 0.976 * 0.976 * 3 * num_bones * 1e-3
    vol_meat = 0.976 * 0.976 * 3 * num_meat * 1e-3
    vol_fat = 0.976 * 0.976 * 3 * num_fat * 1e-3

    overall_tissue = num_fat + num_meat + num_bones
    percent_bone = round(num_bones / overall_tissue, 2)
    percent_meat = round(num_meat / overall_tissue, 2)
    percent_fat = round(num_fat / overall_tissue, 2)

    return vol_bones, vol_meat, vol_fat, percent_bone, percent_meat, percent_fat