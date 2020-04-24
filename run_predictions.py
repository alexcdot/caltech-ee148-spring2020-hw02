import os
import numpy as np
import json
import csv
from PIL import Image, ImageDraw
import cv2
from matplotlib.pyplot import imshow


def draw_rectangle(I, xy):
    img = Image.fromarray(I)
    draw = ImageDraw.Draw(img)
    draw.rectangle([xy[1], xy[0], xy[3], xy[2]])
    text = str(np.round(xy[4], 2))
    draw.text((xy[1], xy[0] - 10), text)
    del draw
    return img


def downsample(I, factor):
    assert(I.shape[0] % factor == 0 and I.shape[1] % factor == 0)
    input_size = I.shape[:2]
    output_size = [dim // factor for dim in I.shape[:2]]
    small_image = I.reshape((output_size[0], factor, 
                             output_size[1], factor, 3)).max(3).max(1)
    return small_image


def calc_iou(A, B):
    # Checks if bbox A and B intersect at all
    # A fixed, B intersects top
    summed_area = ((A[2] - A[0]) * (A[3] - A[1]) +
                   (B[2] - B[0]) * (B[3] - B[1]))
    i_area = 0
    if A[1] >= B[1] and A[1] <= B[3]:
        # intersects topleft
        if A[0] >= B[0] and A[0] <= B[2]:
            i_area = (B[2] - A[0]) * (B[3] - A[1])
        # intersects topright
        elif A[2] >= B[0] and A[2] <= B[2]:
            i_area = (A[2] - B[0]) * (B[3] - A[1])
    # B intersects bottom
    elif A[3] >= B[1] and A[3] <= B[3]:
        # intersects bottomleft
        if A[0] >= B[0] and A[0] <= B[2]:
            i_area = (B[2] - A[0]) * (A[3] - B[1])
        # intersects bottomright
        elif A[2] >= B[0] and A[2] <= B[2]:
            i_area = (A[2] - B[0]) * (A[3] - B[1])
    
    return i_area / (summed_area - i_area)

assert(calc_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175)


def run_nms(bboxes, iou_thres=0.5):
    # Assume bboxes' 4 coord is conf
    valid_bboxes = []
    bboxes = sorted(bboxes, key=lambda x:x[4], reverse=True)

    for bbox in bboxes:
        is_valid = True
        for valid_bbox in valid_bboxes:
            if calc_iou(bbox, valid_bbox) > 0.1:
                is_valid = False
                break
        if is_valid:
            valid_bboxes.append(bbox)
    return valid_bboxes


def rgb_to_hsv(I):
    '''
    Turn a numpy RGB image to a numpy HSV image
    '''
    I = I / 255.0
    cmax = np.max(I, axis=2)
    cmin = np.min(I, axis=2)
    diff = cmax - cmin
    max_color = np.argmax(I, axis=2)
    # initialize hsv
    h = np.zeros(I.shape[:2])
    s = np.zeros(I.shape[:2])
    v = cmax * 100
    # calculate h, s
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # saturation
            if cmax[i, j] > 0:
                s[i, j] = diff[i, j] / cmax[i, j] * 100
            # Both are zero
            if cmax[i, j] == cmin[i, j]:
                continue
            # r
            if max_color[i, j] == 0:
                h[i, j] = (60 * ((I[i, j, 1] - I[i, j, 2]) / diff[i, j]) + 360) % 360
            # g
            if max_color[i, j] == 1:
                h[i, j] = (60 * ((I[i, j, 2] - I[i, j, 0]) / diff[i, j]) + 120) % 360
            # b
            if max_color[i, j] == 2:
                h[i, j] = (60 * ((I[i, j, 0] - I[i, j, 1]) / diff[i, j]) + 240) % 360
    # Flip hue so that red is in the 180 band instead of 0, 360 band
    h  = (h + 180) % 360
    """
    print('h:', h[:5, :5])
    print('s:', s[:5, :5])
    print('v:', v[:5, :5])
    print('h:', h)
    print('s:', s)
    print('v:', v)
    if h.shape[0] > 300:
        cv2.imshow('image', h / 2)
        cv2.waitKey(0)
        cv2.imshow('image', s)
        cv2.waitKey(0)
        cv2.imshow('image', v)
        cv2.waitKey(0)
    """
    return np.stack([h, s, v]).transpose([1,2,0]).astype(np.uint8)


def normalize_image(img, source_img=None, axis=None):
    if source_img is None:
        source_img = img
    return (
        (
            img
            # - source_img.mean(axis=axis, keepdims=True)
        ) /
        np.linalg.norm(source_img)
    )


def process_filters(img, read_only=False):
    if read_only:
        img = img.copy()
    hsv = rgb_to_hsv(img)
    #img = np.concatenate((img, hsv), axis=2)
    # only care about hsv
    img = hsv[:, :, :3]
    return img


def imshow_hsv(I):
    cv2_hsv = I.copy()
    cv2_hsv[:,:,0] /= 2
    cv2_hsv[:,:,1] *= 2.55
    cv2_hsv[:,:,2] *= 2.55
    cv2_hsv = np.round(cv2_hsv).astype(np.uint8)
    #print('viewed h:', cv2_hsv[:5, :5, 0])
    #print('viewed s:', cv2_hsv[:5, :5, 1])
    #print('viewed v:', cv2_hsv[:5, :5, 2])
    #print('viewed hsv:', cv2_hsv, cv2_hsv.shape, cv2_hsv.dtype)
    #proper_hsv = cv2.cvtColor(original_I, cv2.COLOR_RGB2HSV)
    #print("proper hsv:", proper_hsv, proper_hsv.shape, proper_hsv.dtype)
    #print(cv2_hsv.shape)
    cv2.imshow('image', cv2.cvtColor(cv2_hsv, cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)
    """cv2.imshow('image', cv2_hsv[:, :, 0])
    cv2.waitKey(0)
    cv2.imshow('image', cv2_hsv[:, :, 1])
    cv2.waitKey(0)
    cv2.imshow('image', cv2_hsv[:, :, 2])
    cv2.waitKey(0)"""


class Filter:
    def __init__(self, filepath, thres, sourcepath, factor=2):
        self.factor = factor
        rgb_source_I = np.asarray(Image.open(os.path.join(sourcepath)))
        rgb_source_I = downsample(rgb_source_I, self.factor)
        self.source_I = process_filters(rgb_source_I, read_only=True)
        rgb_ref_I = np.asarray(Image.open(os.path.join(filepath)))
        rgb_ref_I = downsample(rgb_ref_I, self.factor)
        self.ref_I = normalize_image(process_filters(rgb_ref_I, read_only=True))
        self.ref_vec = self.ref_I.flatten()
        print(self.ref_I.shape, self.source_I.mean())
        self.ref_rows, self.ref_cols = self.ref_I.shape[:2]
        self.thres = thres


    def get_detections(self, I):
        # Assume I is already downsampled and processed
        # run convolution
        n_rows, n_cols = I.shape[:2]
        bboxes = []
        confs = []
        print(I.mean())
        ind = 0
        for row in range(0, int(n_rows * 0.7) - self.ref_rows, max(self.ref_rows // 10, 1)):
            for col in range(0, n_cols - self.ref_cols, max(self.ref_cols // 10, 1)):
                ind += 1
                if ind % 1 != 0:
                    continue
                sub_mat = I[row:row+self.ref_rows,
                            col:col+self.ref_cols]
                # sub_mat = process_filters(sub_mat)
                sub_vec = normalize_image(sub_mat.flatten())
                conf = np.dot(sub_vec, self.ref_vec)
                confs.append(conf)
                if conf > self.thres:
                    ind -= 1
                    bboxes.append([
                        row * self.factor, col * self.factor,
                        (row+self.ref_rows) * self.factor,
                        (col+self.ref_cols) * self.factor,
                        conf
                    ])
        
        print(sorted(confs)[-10:])
        print('num boxes:', len(bboxes))
        return bboxes


def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    pass

    '''
    END YOUR CODE
    '''


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    '''
    BEGIN YOUR CODE
    '''
    pass

    '''
    END YOUR CODE
    '''


def detect_red_light_mf(I, det_filters, factor=2):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    bboxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    (n_rows,n_cols,n_channels) = np.shape(I)
    
    original_I = I.copy()
    # downsample
    I = downsample(I, factor)
    I = process_filters(I)

    bboxes = []
    for det_filter in det_filters:
        # Similar to predict_boxes(compute_convolution(I, T))
        bboxes.extend(det_filter.get_detections(I))


    print('# bboxes before nms:', len(bboxes))
    bboxes = run_nms(bboxes)
    print('# bboxes after nms:', len(bboxes))
    bboxes = sorted(bboxes, key=lambda x: x[4])
    print("Best bboxes:\n", bboxes[-10:])

    if len(bboxes) == 0:
        return bboxes

    for bbox in bboxes[-10:]:
        original_I = draw_rectangle(original_I, bbox)
        original_I = np.asarray(original_I)

    cv2.imshow('image', cv2.cvtColor(original_I, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    output = bboxes
    '''
    END YOUR CODE
    '''
    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set the path to load the filters csv:
filters_path = '../data/filters'
filters_info_filename = 'thresholds.csv'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

# get detection filters
det_filters = []
with open(os.path.join(filters_path, filters_info_filename)) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    # next(reader)
    # next(reader)
    for row in reader:
        det_filters.append(
            Filter(
                os.path.join(filters_path, row[0]),
                float(row[1]),
                os.path.join(filters_path, row[2]),
            ))
'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    file_names_train[i] = 'RL-062.jpg'
    print('file name:', file_names_train[i])
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, det_filters)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
