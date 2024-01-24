import numpy
import numpy as np
import os
import tensorflow as tf
import pathlib
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# while "models" in pathlib.Path.cwd().parts:
#     os.chdir('..')

test_save_path = ""

Config_path = "config.txt"
config = open(Config_path).read()
config = config.splitlines()
use_cpu = config[4]
thresh = float(config[5])
if use_cpu == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PATH_TO_MODEL_DIR = config[0]
PATH_TO_LABELS = config[1]
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR

# загрузка модели
#print("lol")

detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

def horizontal_split(img, output_dict):
    rows, cols, ch = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    splitline = int(rows/2)
    flag = True
    while flag:
        flag = False
        for box in output_dict['detection_boxes']:
            #box[y1,x1,y2,x2]
            y1 = int(box[0]*rows)
            y2 = int(box[2]*rows)
            #print(str(y1) + " _ "+ str(y2) +" _ "+ str(splitline))
            if splitline > y1 and splitline < y2:
                #box_square = square(box[1],box[0],box[3],box[2])
                #cut_square = square(box[1], box[0], box[3], splitline/rows)
                #if box_square/cut_square < 20:
                splitline -= 4
                flag = True
                # cropped_image1 = img[0:splitline, 0:cols]
                # im = Image.fromarray(cropped_image1)
                # im = im.convert('RGB')
                # im.save(test_save_path +str(img[0][0])+ str(splitline) +".jpg")

    if splitline < rows/3:
        splitline = rows-1
    cropped_image1 = img[0:splitline, 0:cols]
    cropped_image2 = img[splitline:rows, 0:cols]


        #     flag = False
        # if splitlineTop < rows/4:
        #     return [], []
    return cropped_image1, cropped_image2




def affine_transform(img):
    rows, cols, ch = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    # camera is located not perpendicular to frame, so this code transforms image to somewhat straight
    # this improves detection quality
    pts1 = numpy.float32([[0, rows * 0.05], [cols * 1.05, rows * 0.05], [cols * 0.1, rows * 0.98], [cols * 0.95, rows]])
    pts2 = numpy.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    first_third = int(cols / 3)
    second_third = cols - first_third
    cropped_image1 = dst[0:rows, 0:first_third]
    cropped_image2 = dst[0:rows, first_third:second_third]
    cropped_image3 = dst[0:rows, second_third:cols]
    return cropped_image1, cropped_image2, cropped_image3

def affine_transform2(img):
    rows, cols, ch = img.shape
    # camera is located not perpendicular to frame, so this code transforms image to somewhat straight
    # this improves detection quality
    pts1 = numpy.float32([[0, rows * 0.05], [cols * 1.05, rows * 0.05], [cols * 0.1, rows * 0.98], [cols * 0.95, rows]])
    pts2 = numpy.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    first_third = int(cols / 3)
    second_third = cols - first_third
    half = int(rows/2)
    cropped_image1 = dst[0:half, 0:first_third]
    cropped_image2 = dst[0:half, first_third:second_third]
    cropped_image3 = dst[0:half, second_third:cols]
    cropped_image4 = dst[half:rows, 0:first_third]
    cropped_image5 = dst[half:rows, first_third:second_third]
    cropped_image6 = dst[half:rows, second_third:cols]
    return cropped_image1, cropped_image2, cropped_image3, cropped_image4, cropped_image5, cropped_image6

def distance(x1, y1, x2, y2):
    return np.sqrt(((x1-x2)**2 + (y1-y2)**2))

def square(x1, y1, x2, y2):
    y = abs(y2-y1)
    x = abs(x2-x1)
    return x*y


def dumb_detection(img):
    image, detections = show_inference(detection_model, img)
    return image

def delete_detection(output_dict, i):
    output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], i, 0)
    output_dict['raw_detection_boxes'] = np.delete(output_dict['raw_detection_boxes'], i, 0)
    output_dict['raw_detection_scores'] = np.delete(output_dict['raw_detection_scores'], i, 0)
    output_dict['detection_multiclass_scores'] = np.delete(output_dict['detection_multiclass_scores'], i, 0)
    output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], i, 0)
    output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], i, 0)
    output_dict['detection_anchor_indices'] = np.delete(output_dict['detection_anchor_indices'], i, 0)
    return output_dict

def remove_overlap(output_dict):
    threshold = 0.12
    global thresh
    output_dict1 = output_dict.copy()
    i = 0
    kek = 0
    # for scores in output_dict['detection_multiclass_scores']:
    #     for l, score in enumerate(scores):
    #         if score < max(scores):
    #             scores[l] = 0
    # for scores in output_dict['raw_detection_scores']:
    #     for l, score in enumerate(scores):
    #         if score < max(scores):
    #             scores[l] = 0

    while i < output_dict['detection_classes'].size-1:
        if output_dict['detection_scores'][i] < thresh:
            output_dict = delete_detection(output_dict, i)
            continue
        #print(i)
        #new_scores = np.array()
        #print(output_dict['detection_multiclass_scores'])
        #print(output_dict['detection_multiclass_scores'])
        j = output_dict['detection_classes'].size - 1
        while j > i:
            #print(j)
            #print("i = " + str(i) + " j = " + str(j) + " | " + str(output_dict['detection_classes'].size) + " | " + str(output_dict['raw_detection_boxes'][i]))
            x1, y1, x11, y11 = output_dict['detection_boxes'][i]
            x2, y2, x22, y22 = output_dict['detection_boxes'][j]
            dist1 = distance(x1, y1, x2, y2)
            dist2 = distance(x11, y11, x22, y22)
            if (dist1 + dist2) < threshold:
                #print("dist:")
                #print(dist2 + dist1)
                kek+=1
                if output_dict['detection_scores'][i] < output_dict['detection_scores'][j]:
                    output_dict = delete_detection(output_dict, i)
                else:
                    output_dict = delete_detection(output_dict, j)
            j -= 1
        i += 1
    return output_dict


# def detect_and_count(img):
#     global thresh
#
#     # model efficientdetd0 can only process images of size 512x512,
#     # so I split large input image into 3 parts and process them separately
#     # due to specific of task it doesn't create error in detections
#     cropped1, cropped2, cropped3 = affine_transform(img)
#     image1, detections1 = show_inference(detection_model, cropped1)
#     image2, detections2 = show_inference(detection_model, cropped2)
#     image3, detections3 = show_inference(detection_model, cropped3)
#     dicts = [detections1, detections2, detections3]
#     image = cv2.hconcat([image1, image2, image3])
#     detection_numbers = dict()
#     for detections in dicts:
#         for i in range(0, detections['detection_classes'].size):
#             if detections['detection_scores'][i] >= thresh:
#                 if detection_numbers.get(detections['detection_classes'][i]) is None:
#                     detection_numbers.setdefault(detections['detection_classes'][i], 1)
#                 else:
#                     detection_numbers[detections['detection_classes'][i]] += 1
#     return image, detection_numbers, detections['detection_classes'].size

def merge_dict(dict1, dict2):
    numpy.append(dict1['detection_classes'],dict2['detection_classes'])
    numpy.append(dict1['raw_detection_boxes'],dict2['raw_detection_boxes'])
    numpy.append(dict1['detection_multiclass_scores'],dict2['detection_multiclass_scores'])
    numpy.append(dict1['detection_boxes'],dict2['detection_boxes'])
    numpy.append(dict1['detection_scores'],dict2['detection_scores'])
    numpy.append(dict1['detection_anchor_indices'],dict2['detection_anchor_indices'])
    return dict1
def detect_and_count(img):
    global thresh

    # model efficientdetd0 can only process images of size 512x512,
    # so I split large input image into 3 parts and process them separately
    # due to specific of task it doesn't create error in detections
    cropped1, cropped2, cropped3 = affine_transform(img)
    image1, detections1 = show_inference(detection_model, cropped1)
    image2, detections2 = show_inference(detection_model, cropped2)
    image3, detections3 = show_inference(detection_model, cropped3)
    detections1 = remove_overlap(detections1)
    detections2 = remove_overlap(detections2)
    detections3 = remove_overlap(detections3)
    cropped1h1, cropped1h2 = horizontal_split(cropped1, detections1)
    cropped2h1, cropped2h2 = horizontal_split(cropped2, detections2)
    cropped3h1, cropped3h2 = horizontal_split(cropped3, detections3)

    if len(cropped1h1) > 1 and len(cropped1h2) > 1 :
        image1h1, detections1h1 = show_inference(detection_model, cropped1h1)
        image1h2, detections1h2 = show_inference(detection_model, cropped1h2)
        image1 = cv2.vconcat([image1h1, image1h2])
        detections1 = merge_dict(detections1h1,detections1h2)
    if len(cropped2h1) > 1 and len(cropped2h2) > 1 :
        image2h1, detections2h1 = show_inference(detection_model, cropped2h1)
        image2h2, detections2h2 = show_inference(detection_model, cropped2h2)
        image2 = cv2.vconcat([image2h1, image2h2])
        detections2 = merge_dict(detections2h1, detections2h2)
    if len(cropped3h1) > 1 and len(cropped3h2) > 1 :
        image3h1, detections3h1 = show_inference(detection_model, cropped3h1)
        image3h2, detections3h2 = show_inference(detection_model, cropped3h2)
        image3 = cv2.vconcat([image3h1, image3h2])
        detections3 = merge_dict(detections3h1, detections3h2)

    dicts = [detections1, detections2, detections3]
    image = cv2.hconcat([image1, image2, image3])
    detection_numbers = dict()
    for detections in dicts:
        for i in range(0, detections['detection_classes'].size):
            if detections['detection_scores'][i] >= thresh:
                if detection_numbers.get(detections['detection_classes'][i]) is None:
                    detection_numbers.setdefault(detections['detection_classes'][i], 1)
                else:
                    detection_numbers[detections['detection_classes'][i]] += 1
    return image, detection_numbers, detections['detection_classes'].size


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, frame):
    # take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    image_np=np.compress([True, True, True, False], image_np, axis=2)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    output_dict = remove_overlap(output_dict)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        max_boxes_to_draw=400,
        min_score_thresh=thresh,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=5)

    return (image_np, output_dict)

