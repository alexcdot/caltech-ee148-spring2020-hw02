import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    if box_1[1] < box_2[1]:
        upper = box_1
        lower = box_2
    else:
        upper = box_2
        lower = box_1
        
    if box_1[0] < box_2[0]:
        left = box_1
        right = box_2
    else:
        left = box_2
        right = box_1
    summed_area = ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]) +
                   (box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
    if upper[3] - lower[1] <= 0 or left[2] - right[0] <= 0:
        iou = 0
    else:
        intersecting_area = (upper[3] - lower[1]) * (left[2] - right[0])
        iou = intersecting_area / (summed_area - intersecting_area)
    assert (iou >= 0) and (iou <= 1.0)
    return iou

assert(compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175)
assert(compute_iou([0, 0, 10, 10], [11, 11, 15, 15]) == 0)


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0
    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        matched_preds = set()
        FP += len(pred)
        FN += len(gt)
        for i in range(len(gt)):
            pred_matched = False
            for j in range(len(pred)):
                if j in matched_preds:
                    continue
                iou = compute_iou(pred[j][:4], gt[i])
                # print(i, j, iou)
                if iou >= iou_thr and pred[j][4] >= conf_thr:
                    TP += 1
                    FP -= 1
                    FN -= 1
                    pred_matched = True
                    matched_preds.add(j)
                    break
    '''
    END YOUR CODE
    '''
    return TP, FP, FN

assert(
    compute_counts(
        {'a': [[0, 0, 10, 10, 1], [5, 5, 10, 10, 1], [11, 11, 21, 21, 1]]},
        {'a': [[0, 0, 10, 10], [12, 12, 22, 22], [20, 20, 30, 30]]}
    ) == (2,1,1)
)

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

iou_thrs = [0.25, 0.5, 0.75]

# using (ascending) list of confidence scores as thresholds
confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname
    in preds_train],dtype=float))

# tp, fp, fn, pr, rc
train_metrics = np.zeros((len(iou_thrs), len(confidence_thrs), 5))
test_metrics = np.zeros((len(iou_thrs), len(confidence_thrs), 5))

for i, iou_thr in enumerate(iou_thrs):
    for j, conf_thr in enumerate(confidence_thrs):
        train_metrics[i, j, :2] = compute_counts(
            preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

train_metrics[:, :, 3] = (
    train_metrics[:, :, 0] / (train_metrics[:, :, 0] + train_metrics[:, :, 1])
)
train_metrics[:, :, 4] = (
    train_metrics[:, :, 0] / (train_metrics[:, :, 0] + train_metrics[:, :, 2])
)

np.savetxt(os.path.join(preds_path, 'train_metrics.txt'), train_metrics)

for i, iou_thr in enumerate(iou_thrs):
    plt.plot(train_metrics[:, i, 4], train_metrics[:, i, 3],
             label='IOU threshold: ' + str(iou_thr))

plt.title('Train PR curves')
plt.legend()
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.savefig('train_pr_curve.png')
plt.show()
# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
        
    for i, iou_thr in enumerate(iou_thrs):
        for j, conf_thr in enumerate(confidence_thrs):
            test_metrics[i, j] = compute_counts(
                preds_test, gts_test, iou_thr=iou_thr, conf_thr=conf_thr)

    test_metrics[:, :, 3] = (
        test_metrics[:, :, 0] / (test_metrics[:, :, 0] + test_metrics[:, :, 1])
    )
    test_metrics[:, :, 4] = (
        test_metrics[:, :, 0] / (test_metrics[:, :, 0] + test_metrics[:, :, 2])
    )

    np.savetxt(os.path.join(preds_path, 'test_metrics.txt'), test_metrics)

    for i, iou_thr in enumerate(iou_thrs):
        plt.plot(test_metrics[:, i, 4], test_metrics[:, i, 3],
                label='IOU threshold: ' + str(iou_thr))

    plt.title('Test PR curves')
    plt.legend()
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig('test_pr_curve.png')
    plt.show()