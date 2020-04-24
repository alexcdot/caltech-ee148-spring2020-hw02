import os
import numpy as np
import json
import csv
from PIL import Image, ImageDraw
import cv2


def draw_rectangle(I, xy, color='white'):
    img = Image.fromarray(I)
    draw = ImageDraw.Draw(img)
    draw.rectangle([xy[1], xy[0], xy[3], xy[2]], outline=color)
    del draw
    return img
"""    PIL.ImageDraw.Draw.rectangle(xy, fill=None, outline=None)
    Draws a rectangle.

    Parameters:	
    xy – Four points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
    outline – Color to use for the outline.
    fill – Color to use for the fill.
"""


def visualize_bboxes(img, gt_bboxes, pred_bboxes, filepath=None):
    for bbox in pred_bboxes[-10:]:
        img = draw_rectangle(img, bbox, 'red')
        img = np.asarray(img)
    for bbox in gt_bboxes[-10:]:
        img = draw_rectangle(img, bbox, 'green')
        img = np.asarray(img)
    if filepath is not None:
        print(filepath)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #print(img.shape)
    #cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)



if __name__ == "__main__":
    save_path = '../data/hw02_preds'
    """    preds = json.load(open('../data/hw02_preds/preds_train.json'))
    gt = json.load(open('../data/hw02_annotations/formatted_annotations_students.json'))
    for filename in preds:
        img = Image.open(os.path.join('../data/RedLights2011_Medium', filename))
        img = np.asarray(img)
        visualize_bboxes(img, gt[filename], preds[filename],
            os.path.join(save_path, 'train_pred_'+filename))
    """
    done_tweaking = True
    if done_tweaking:
        preds = json.load(open('../data/hw02_preds/preds_test.json'))
        gt = json.load(open('../data/hw02_annotations/formatted_annotations_students.json'))
        for filename in preds:
            img = Image.open(os.path.join('../data/RedLights2011_Medium', filename))
            img = np.asarray(img)
            visualize_bboxes(img, gt[filename], preds[filename],
                os.path.join(save_path, 'test_pred_'+filename))
