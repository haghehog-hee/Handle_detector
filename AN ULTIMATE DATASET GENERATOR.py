import os
import cv2
import numpy as np
from autoannotations import detection_to_text
import math

backgrounds_dir = "C:\\Tensorflow\\Dataset\\greenscreen\\backgrounds\\"
handle_dir = "C:\\Tensorflow\\Dataset\\greenscreen\\edited\\"
save_dir = "C:\\Tensorflow\\Dataset\\greenscreen\\result\\"
ANNOTATION_SAVE_PATH = "C:\\Tensorflow\\Dataset\\greenscreen\\annotations\\"
handle_dirs = os.listdir(handle_dir)

num_pictures_to_generate = 300
background_paths = os.listdir(backgrounds_dir)
iou_threshold = 0.5

def randdir(dir):
    num = int(np.random.rand()*len(dir))
    result = dir[num]
    return result

def randimg(dir):
    paths = os.listdir(dir)
    x = np.random.rand()
    x = int(x * len(paths)-1)
    # print(len(paths))
    # print(x)
    num = x
    path = dir + paths[num]
    img = cv2.imread(path)
    return img, path


def randimg2(dir):
    paths = os.listdir(dir)
    x = np.random.rand()
    x = int(x * len(paths)-1)
    # print(len(paths))
    # print(x)
    num = x
    path = dir + paths[num]
    img = cv2.imread(path, -1)
    return img

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def random_flip(img):
    n = np.random.rand()
    d = np.random.rand()
    if n > 0.5:
        if d > 0.5:
            img = cv2.flip(img, 1)
        else:
            img = cv2.flip(img, 0)
    return img

def add_row(large_img, small_img, x_offset, y_offset, boxes, detections):
    h_height = small_img.shape[0]
    h_width = small_img.shape[1]
    flag = 1
    row_size = int(np.random.rand() * 10 + 1)
    angle = np.random.rand() * 2 * math.pi
    offset = np.random.rand() * small_img.shape[0]
    offset_y = int(offset * math.sin(angle))
    offset_x = int(offset * math.cos(angle))
    for i in range(row_size):
        x_offset += offset_x
        y_offset += offset_y
        if x_offset + h_width >= large_img.shape[1] or y_offset + h_height >= large_img.shape[0] or x_offset < 0 or y_offset < 0:
            return large_img, flag, boxes, detections
        large_img, flag, boxes, detections = add_img(large_img, small_img, x_offset, y_offset, boxes, detections)
    return large_img, flag, boxes, detections
    print("lol")
def add_img(large_img, small_img, x_offset, y_offset, boxes, detections):
    print("lol")
    s_img = small_img
    l_img = large_img
    h_height = s_img.shape[0]
    h_width = s_img.shape[1]
    y1, y2 = y_offset, y_offset + h_height
    x1, x2 = x_offset, x_offset + h_width
    # print(s_img.shape)
    # print(l_img.shape)
    # print(x_offset)
    # print(y_offset)
    new_box = [y_offset, x_offset, (y_offset + h_height), (x_offset + h_width)]
    for box in boxes:
        iou = bb_intersection_over_union(new_box, box)

        if iou > iou_threshold:
            return l_img, False, boxes, detections
    boxes.append(new_box)
    detections['detection_scores'].append(1)
    detections['detection_classes'].append(handle_num)
    detections['detection_boxes'].append(new_box)

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        lol = l_img[y1:y2, x1:x2, c]
        kek = alpha_l * lol
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + kek )
    return l_img, True, boxes, detections

for i in range(num_pictures_to_generate):
    background, bg_path = randimg(backgrounds_dir)
    height, width, ch  = background.shape

    detections = dict(detection_scores = [], detection_boxes=[], detection_classes = [])

    boxes = []
    flag = True
    crowd_counter = 0
    while flag:
        crowd_flag = False
        handle_path = randdir(handle_dirs)
        handle_num = int(handle_path)
        handle_path = handle_dir + handle_path + "\\"
        handle = randimg2(handle_path)
        handle = random_flip(handle)

        h_width, h_height, ch = handle.shape
        diff = width/h_width
        scale = ((10+np.random.rand()*10)/100)*diff
        handle = cv2.resize(src = handle, dsize = None, fx = scale, fy = scale)
        h_height, h_width, ch = handle.shape
        y_offset = int(np.random.rand()*height)
        x_offset = int(np.random.rand()*width)
        if y_offset + h_height > height:
            y_offset = height - h_height - 1
        if x_offset + h_width > width:
            x_offset= width - h_width - 1

        check = np.random.rand()
        if check < 0.6:
            background, is_added, boxes, detections = add_img(background, handle, x_offset, y_offset, boxes, detections)
        else:
            background, is_added, boxes, detections = add_row(background, handle, x_offset, y_offset, boxes, detections)
        if not is_added:
            crowd_counter += 1
        # new_box = [y_offset, x_offset, (y_offset+h_height), (x_offset+h_width)]
        # for box in boxes:
        #     iou = bb_intersection_over_union(new_box, box)
        #
        #     if iou > iou_threshold:
        #         print(iou)
        #         crowd_counter += 1
        #         print("_____________________________________________________________")
        #         print(crowd_counter)
        #         crowd_flag = True
        #         break
        # if crowd_flag:
        #     continue
        if crowd_counter > 5:
            flag = False
            break
        # if not crowd_flag:
        #     boxes.append(new_box)
        #     detections['detection_scores'].append(1)
        #     detections['detection_classes'].append(handle_num)
        #     detections['detection_boxes'].append(new_box)
        #     check = np.random.rand()
        #         if check > 0.75:
        #         background = add_img(background, handle, x_offset, y_offset)
        #     background = add_img(background, handle, x_offset, y_offset)
    sv_path = save_dir + str(i)+".jpg"
    detections['detection_scores'] = np.array(detections['detection_scores'])
    detections['detection_classes'] = np.array(detections['detection_classes'])
    detections['detection_boxes'] = np.array(detections['detection_boxes'])
    detection_to_text(detections = detections, filename = str(i)+".jpg", image_path=sv_path, width = width, height = height, ANNOTATION_SAVE_PATH=ANNOTATION_SAVE_PATH, scale_flag=False)
    cv2.imwrite(sv_path, background)





