import numpy as np 
import os
import matplotlib.pyplot as plt 

cls_names = ["straight", "left", "right", "stop", "nohonk", "crosswalk"]

def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1+area2-inter_area+1e-6)

    return iou

def process_predicts(predict_txt):
    """ process the prediction. 
    Args:
        predict_txt: a txt file of prection result, one row data is
            formated as "img_path x1 y1 x2 y2 cls x1 y1 x2 y2 cls ...\n" 
    Return:
        predicts_dict: a prediction dict containing all the objects information of all test images. 
            {image1: {"obj1": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...], "obj2": ...}
    """
    predict_dict = {}
    with open(predict_txt, "r") as f:
        predict = f.readlines()

    for predict_row in predict:
        img_dict = {}
        
        predict_row = predict_row.rstrip()
        predict_list = predict_row.split(" ") 
        img_name = os.path.split(predict_list[0])[-1]
        obj_info = predict_list[1:]
        obj_info = [int(i) for i in obj_info]
        predict_dict[img_name] = {}
        obj_num = int((len(obj_info)/5))
        for obj_index in range(obj_num):
            x1 = obj_info[5*obj_index]
            y1 = obj_info[5*obj_index + 1]
            x2 = obj_info[5*obj_index + 2]
            y2 = obj_info[5*obj_index + 3]
            cls = obj_info[5*obj_index + 4]
            obj_name = cls_names[cls]
        
            if not obj_name in img_dict.keys():
                img_dict[obj_name] = []
            img_dict[obj_name].append([x1, y1, x2, y2])
        predict_dict[img_name] = img_dict

    return predict_dict

def compute_precision_recall(label_txt, predict_txt, iou1=0.5):
    """compute the precision and recall.
    Args:
        label_txt: a label txt file of objection information, one row data is
            formated as "img_path x1 y1 x2 y2 cls x1 y1 x2 y2 cls ...\n" 
        predict_txt: a prediction txt file, same with label　txt.
    Return:
        pre_rec: precision and recall dict.
            {cls_name1: [precision1, recall1], cls_name2: [precision2, recall2], ...}
    """
    pre_rec = {}
    tp_count = {}
    img_num_per_cls = {}
    img_num_predict = {}
    for cls_name in cls_names:
        pre_rec[cls_name] = [0.0, 0.0]
        tp_count[cls_name] = 0.0
        img_num_per_cls[cls_name] = 1e-3
        img_num_predict[cls_name] = 1e-3
    
    
    label_dict = process_predicts(label_txt)
    predict_dict = process_predicts(predict_txt)

    
    for img_name, obj_label in label_dict.items():
        obj_predict = predict_dict[img_name]
        for obj, coords in obj_predict.items():
            img_num_predict[obj] += len(coords)
        for obj, coords in obj_label.items():
            img_num_per_cls[obj] += len(coords)
            if obj in obj_predict.keys():
                for coord1 in coords:
                    for coord2 in obj_predict[obj]:
                        if compute_iou(coord1, coord2)>=iou1:
                            tp_count[obj] +=1

    for cls_name in cls_names:
        pre_rec[cls_name] = [tp_count[cls_name]/img_num_predict[cls_name], tp_count[cls_name]/img_num_per_cls[cls_name]]
    
    return pre_rec


if __name__ == "__main__":
    label_txt = "./data/test_images/test.txt"
    predict_txt = "./data/test_result.txt"
    pre_rec = compute_precision_recall(label_txt, predict_txt)
    print(pre_rec)
     
                        



