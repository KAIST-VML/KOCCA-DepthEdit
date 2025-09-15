import os,sys

from tqdm import tqdm
from util import util

import imageio
import scipy
import skimage.color
import numpy as np
from PIL import Image
from scipy.sparse.linalg import spsolve
def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
	
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
	
	return output

# if __name__ == '__main__':
#     root='/data2/kitti_dataset/'
#     list_path = os.path.join(root, "eigen_test_files_with_gt.txt")

#     with open(list_path, 'r') as f:
#         filenames = f.readlines()
		
#         data_rgb = []
#         data_depth = []

#         for line in filenames:
#             names = line.split()
#             if names[1] != "None":
#                 image_name = os.path.join(root, names[0])
#                 depth_name = os.path.join(root,'data_depth_annotated/',names[1])

#                 if os.path.exists(image_name) and os.path.exists(depth_name):
#                     data_rgb.append(image_name)
#                     data_depth.append(depth_name)

#     print("(KITTIDataset)   Total of {} files".format(len(data_rgb)))

#     for index in tqdm(range(len(data_rgb))):
#         rgb = np.array(imageio.imread(data_rgb[index], pilmode="RGB"))
#         depth_png = np.array(imageio.imread(data_depth[index]), dtype=int)
#         depth = depth_png.astype(np.float) / 256.0
		
#         depth_filled =  fill_depth_colorization(rgb, depth, alpha=1)*256

#         filename = data_depth[index].replace("data_depth_annotated", "data_depth_filled")
#         dirname, basename = os.path.split(filename)
#         os.makedirs(dirname, exist_ok=True)
#         util.write_depth(filename.replace(".png",""), depth_filled, bits=2)

if __name__ == '__main__':
	root='/data2/kitti_2015/training'
	depth_names = ['depth_noc_0', 'depth_noc_1', 'depth_occ_0', 'depth_occ_1']
	import glob
	for depth_name in depth_names:
		depth_path = os.path.join(root, depth_name)
		file_list_im = os.listdir(depth_path)
		print("(KITTIDataset)   Total of {} files".format(len(file_list_im)))

		for index in tqdm(range(len(file_list_im))):
			depth_file = os.path.join(depth_path, file_list_im[index])
			rgb_file = depth_file.replace(depth_name, "image_2")
			rgb = np.array(imageio.imread(rgb_file, pilmode="RGB"))
			depth_png = np.array(imageio.imread(depth_file), dtype=int)
			depth = depth_png.astype(np.float) / 256.0
			
			depth_filled =  fill_depth_colorization(rgb, depth, alpha=1)*256

			filename = depth_file.replace(depth_name, "data_depth_filled")
			dirname, basename = os.path.split(filename)
			os.makedirs(dirname, exist_ok=True)
			util.write_depth(filename.replace(".png",""), depth_filled, bits=2,absolute_depth=True)
