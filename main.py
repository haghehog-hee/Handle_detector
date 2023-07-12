import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')



PATH_TO_MODEL_DIR = "C:\\Tensorflow\\models\\research\\object_detection\\200objects\\saved_model"
PATH_TO_LABELS = "C:\\Tensorflow\\Dataset\\label_map.pbtxt"

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_inception_v2_coco_2017_11_17'


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
        # Reframe the the bbox mask to the image size.
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
    #input_tensor = tf.convert_to_tensor(image_np)
   #input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        max_boxes_to_draw=400,
        min_score_thresh=.2,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=5)
    # detections = model(input_tensor)

    # # конвертирование тензоров в массив numpy и удаление пакета
    # num_detections = int(detections.pop('num_detections'))
    # detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    # detections['num_detections'] = num_detections
    #
    # # визуализация результатов предсказания на изображении
    # image_np_with_detections = image_np.copy()
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_detections,
    #     detections['detection_boxes'],
    #     detections['detection_classes'],
    #     detections['detection_scores'],
    #     category_index,
    #     use_normalized_coordinates=True,
    #     max_boxes_to_draw=400,
    #     min_score_thresh=.2,
    #     agnostic_mode=False)
    return (image_np, output_dict)

def Affine(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[9, 87], [1960, 111], [216, 1104], [1752, 1049]])
    pts2 = np.float32([[0, 0], [1890, 0], [0, 1300], [1890, 1100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (1890, 1100))
    return (dst)