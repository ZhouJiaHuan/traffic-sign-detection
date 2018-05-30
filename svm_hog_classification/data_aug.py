import cv2
import numpy as np
import os

def affine(img, delta_pix):
    rows, cols, _ = img.shape
    pts1 = np.float32([[0,0], [rows,0], [0, cols]])
    pts2 = pts1 + delta_pix
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (rows, cols))
    return res


def affine_dir(img_dir, write_dir, max_delta_pix):
    img_names = os.listdir(img_dir)
    img_names = [img_name for img_name in img_names if img_name.split(".")[-1]=="jpg"]
    for index, img_name in enumerate(img_names):
        img = cv2.imread(os.path.join(img_dir,img_name))
        save_name = os.path.join(write_dir, img_name.split(".")[0]+"f.jpg")
        delta_pix = np.float32(np.random.randint(-max_delta_pix,max_delta_pix+1,[3,2]))
        img_a = affine(img, delta_pix)
        cv2.imwrite(save_name, img_a)

if __name__ == "__main__":
    img_dir = "/home/meringue/Documents/traffic_sign_detection/data/proposals_train"
    write_dir = "/home/meringue/Documents/traffic_sign_detection/data/proposal_train_affine"
    affine_dir(img_dir, write_dir, 10)