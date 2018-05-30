"""
produce positive and negative samples for classification.
author: Meringue
date: 2018/05/29
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2


classes_name = ["straight", "left", "right", "stop", "nohonk", "crosswalk", "background"]
classes_num = {"straight": 0, "left": 1, "right": 2, "stop": 3, "nohonk": 4, "crosswalk": 5, "background": 6}

SIGN_ROOT = "/home/meringue/Documents/traffic_sign_detection"
DATA_PATH = os.path.join(SIGN_ROOT, 'data/images/')
OUTPUT_PATH = os.path.join(SIGN_ROOT, 'data/images_proposals.txt')



def parse_xml(xml_file):
	"""parse xml_file
	Args:
		xml_file: the input xml file path
	Returns:
    	image_path: string
    	labels: list of [xmin, ymin, xmax, ymax, class]
  	"""
	tree = ET.parse(xml_file)
	root = tree.getroot()
	image_path = ''
	labels = []

	for item in root:
		if item.tag == 'filename':
			image_path = os.path.join(DATA_PATH, "JPEGImages/", item.text)
		elif item.tag == 'object':
			obj_name = item[0].text
			obj_num = classes_num[obj_name]
			xmin = int(item[4][0].text)
			ymin = int(item[4][1].text)
			xmax = int(item[4][2].text)
			ymax = int(item[4][3].text)
			labels.append([xmin, ymin, xmax, ymax, obj_num])
  	return image_path, labels


def produce_neg_proposals(img_path, write_dir, min_size, square=False, proposal_num=0):
	"""produce negative proposals from a negative image.
	Args:
		img_path: image path.
		write_dir: write directory.
		min_size: the minimum size of the proposals.
		square:  crop a square or not.
		proposal_num: current negative proposal numbers.
	Return:
		proposal_num: negative proposal numbers.
	"""
	img = cv2.imread(img_path)
	rows = img.shape[0]
	cols = img.shape[1]
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	imgBinBlue = cv2.inRange(imgHSV,np.array([100,43,46]), np.array([124,255,255]))
	imgBinRed1 = cv2.inRange(imgHSV,np.array([0,43,46]), np.array([10,255,255]))
	imgBinRed2 = cv2.inRange(imgHSV,np.array([156,43,46]), np.array([180,255,255]))
	imgBinRed = np.maximum(imgBinRed1, imgBinRed2)
	imgBin = np.maximum(imgBinRed, imgBinBlue)

	_, contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for contour in contours:
		x,y,w,h = cv2.boundingRect(contour)
		if w<min_size or h<min_size:
			continue

		if square is True:
			xcenter = int(x+w/2)
			ycenter = int(y+h/2)
			size = max(w,h)
			xmin = max(xcenter-size/2, 0)
			xmax = min(xcenter+size/2,cols)
			ymin = max(ycenter-size/2, 0)
			ymax = min(ycenter+size/2,rows)
			proposal = img[ymin:ymax, xmin:xmax]
			proposal = cv2.resize(proposal, (size,size))

		else:
			proposal = img[y:y+h, x:x+w]
		write_name = "background" + "_" + str(proposal_num) + ".jpg"
		proposal_num += 1
		cv2.imwrite(os.path.join(write_dir,write_name), proposal)
	return proposal_num

def produce_pos_proposals(img_path, write_dir, labels, min_size, square=False, proposal_num=0, ):
	"""produce positive proposals based on labels.
	Args:
		img_path: image path.
		write_dir: write directory.
		min_size: the minimum size of the proposals.
		labels: a list of bounding boxes.
			[[x1, y1, x2, y2, cls_num], [x1, y1, x2, y2, cls_num], ...]
		square:  crop a square or not.
	Return:
		proposal_num: proposal numbers.
	"""
	img = cv2.imread(img_path)
	rows = img.shape[0]
	cols = img.shape[1]
	for label in labels:
		xmin, ymin, xmax, ymax, cls_num = np.int32(label)
		if xmax-xmin<min_size or ymax-ymin<min_size:
			continue
		if square is True:
			xcenter = int((xmin + xmax)/2)
			ycenter = int((ymin + ymax)/2)
			size = max(xmax-xmin, ymax-ymin)
			xmin = max(xcenter-size/2, 0)
			xmax = min(xcenter+size/2,cols)
			ymin = max(ycenter-size/2, 0)
			ymax = min(ycenter+size/2,rows)
			proposal = img[ymin:ymax, xmin:xmax]
			proposal = cv2.resize(proposal, (size,size))
		else:
			proposal = img[ymin:ymax, xmin:xmax]
				
		cls_name = classes_name[cls_num]
		proposal_num[cls_name] +=1
		write_name = cls_name + "_" + str(proposal_num[cls_name]) + ".jpg"
		cv2.imwrite(os.path.join(write_dir,write_name), proposal)
	return proposal_num



def produce_proposals(xml_dir, write_dir, square=False, min_size=30):
	"""produce proposals (positive examples for classification) to disk.
	Args:
    	xml_dir: image xml file directory.
		write_dir: write directory of all proposals.
		square: crop a square or not.
		min_size: the minimum size of the proposals.
	Returns:
		proposal_num: a dict of proposal numbers.
	"""

	proposal_num = {}
	for cls_name in classes_name:
		proposal_num[cls_name] = 0

	index = 0
	for xml_file in os.listdir(xml_dir):
		img_path, labels = parse_xml(os.path.join(xml_dir,xml_file))
		img = cv2.imread(img_path)
		rows = img.shape[0]
		cols = img.shape[1]

		if len(labels) == 0:
			neg_proposal_num = produce_neg_proposals(img_path, write_dir, min_size, square, proposal_num["background"])
			proposal_num["background"] = neg_proposal_num
		else:
			proposal_num = produce_pos_proposals(img_path, write_dir, labels, min_size, square=True, proposal_num=proposal_num)
			
		if index%100 == 0:
			print "total xml file number = ", len(os.listdir(xml_dir)), "current xml file number = ", index
			print "proposal num = ", proposal_num
		index += 1

	return proposal_num



if __name__ == "__main__":
	xml_dir = "/home/meringue/Documents/traffic_sign_detection/data/train_images/Annotations"
	save_dir = "/home/meringue/Documents/traffic_sign_detection/data/proposals_train"
	proposal_num = produce_proposals(xml_dir, save_dir, square=True)
	print "proposal num = ", proposal_num

