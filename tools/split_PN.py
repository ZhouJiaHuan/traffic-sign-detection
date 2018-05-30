"""
split data to positive and negative and save to specified directory according
to the positive annotations.
author: Meringue
date: 2018/05/29
"""

import os
import shutil
import random

def _copy_file(src_file, dst_file):
	"""copy file.
	"""
	if not os.path.isfile(src_file):
		print"%s not exist!" %(src_file)
	else:
		fpath, fname = os.path.split(dst_file)
		if not os.path.exists(fpath):
			os.makedirs(fpath)
		shutil.copyfile(src_file, dst_file)

def splitPN(img_dir, pos_dir, neg_dir, pos_anno_dir):
	""" split to positive and negtative images .
	Args:
		img_dir: all image directory to to be splitted.
		neg_dir, pos_dir: splitted directory.
		pos_anno_dir: positive annotations directory.
	"""
	all_names = os.listdir(img_dir)
	all_names = [name.split(".")[0] for name in all_names]
	pos_names = os.listdir(pos_anno_dir)
	pos_names = [name.split(".")[0] for name in pos_names]

	for name in all_names:
		if name in pos_names:
			_copy_file(os.path.join(img_dir,name+".jpg"), os.path.join(pos_dir,name+".jpg"))
		else:
			_copy_file(os.path.join(img_dir,name+".jpg"), os.path.join(neg_dir,name+".jpg"))


if __name__ == "__main__":
	img_dir = "/home/meringue/Desktop/JPEGImages"
	pos_dir = "/home/meringue/Desktop/Pos"
	neg_dir = "/home/meringue/Desktop/Neg"
	pos_anno_dir = "/home/meringue/Desktop/Annotations_P" #the annotation directory of positive examples.

	print "start splitting..."
	splitPN(img_dir, pos_dir, neg_dir, pos_anno_dir)

