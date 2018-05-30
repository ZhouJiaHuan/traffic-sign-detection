"""
crop the original image to specified size.
author: Meringue
date: 2018/05/29
"""

import os
import cv2

def center_crop(img_array, crop_size=-1, resize=-1, write_path=None):
	""" crop and resize a square image from the centeral area.
	Args:
		img_array: image array
		crop_size: crop_size (default: -1, min(height, width)).
		resize: resized size (default: -1, keep cropped size)
		write_path: write path of the image (default: None, do not write to the disk).
	Return:
		img_crop: copped and resized image.
	"""
	rows = img_array.shape[0]
	cols = img_array.shape[1]

	if crop_size==-1 or crop_size>max(rows,cols):
		crop_size = min(rows, cols)
	row_s = max(int((rows-crop_size)/2), 0)
	row_e = min(row_s+crop_size, rows) 
	col_s = max(int((cols-crop_size)/2), 0)
	col_e = min(col_s+crop_size, cols)

	img_crop = img_array[row_s:row_e,col_s:col_e,]

	if resize>0:
		img_crop = cv2.resize(img_crop, (resize, resize))

	if write_path is not None:
		cv2.imwrite(write_path, img_crop)
	return img_crop 


def crop_img_dir(img_dir,  save_dir, crop_method = "center", rename_pre=-1):
	""" crop and save square images from original images saved in img_dir.
	Args:
		img_dir: image directory.
		save_dir: save directory.
		crop_method: crop method (default: "center").
		rename_pre: prename of all images (default: -1, use primary image name).
	Return: none
	"""
	img_names = os.listdir(img_dir)
	img_names = [img_name for img_name in img_names if img_name.split(".")[-1]=="jpg"]
	index = 0
	for img_name in img_names:
		img = cv2.imread(os.path.join(img_dir, img_name))

		rename = img_name if rename_pre==-1 else rename_pre+str(index)+".jpg"
		img_out_path = os.path.join(save_dir, rename)

		if crop_method == "center":
			img_crop = center_crop(img, resize=640, write_path=img_out_path)

		if index%100 == 0:
			print "total images number = ", len(img_names), "current image number = ", index
		index += 1
		



if __name__ == "__main__":
	img_dir = "/home/meringue/Documents/traffic_sign_detection/tools/test_images"
	save_dir = "/home/meringue/Documents/traffic_sign_detection/tools/test_out_images"
	crop_img_dir(img_dir, save_dir, rename_pre="sign_")
	

