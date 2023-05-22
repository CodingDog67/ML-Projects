import SimpleITK as sitk
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from segmentation_helper_functions import sample_stack_display

annot_path = "C:\\Users\\Ari\\Desktop\\New folder\\3690 Backfat labelled\\"

label_map = sitk.GetArrayFromImage(sitk.ReadImage(annot_path + "3670 corrected.mhd"))

coords = np.where(label_map == 1)

sample_stack_display(label_map[min(coords[0]):, :, :], "something", display=True, rows=6, cols=6, start_with=10, save=False, labels=True)

label_map = np.flip(label_map, 0)
#undensify by showing every n-th frame
uniframes = np.unique(coords[0])
keep_frames = uniframes[0::5]
delete_slices = [x for x in uniframes if not x in keep_frames]
label_map[delete_slices, :, :] = 0

new_coords = np.where(label_map == 1)
#undensify by showing every n-th row
uniframes = np.unique(new_coords[1])
keep_frames = uniframes[0::10]
delete_slices = [x for x in uniframes if not x in keep_frames]
label_map[:, delete_slices, :] = 0

new_coords = np.where(label_map == 1)
#undensify by showing every n-th col
uniframes = np.unique(new_coords[2])
keep_frames = uniframes[0::10]
delete_slices = [x for x in uniframes if not x in keep_frames]
label_map[:, :, delete_slices] = 0


new_coords = np.where(label_map == 1)

# coords 0 is the frame axis
X = new_coords[0] - min(new_coords[0])
Y = new_coords[1] - min(new_coords[1])
Z = new_coords[2] - min(new_coords[2])  # bring the map down
fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

ax.scatter(X, Y, Z, c=Y, cmap='viridis', linewidth=0.1);
#ax.plot_wireframe(X, Y, Z, color='black')
#ax.set_title('wireframe');

fig.show()