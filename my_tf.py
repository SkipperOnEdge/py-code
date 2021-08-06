import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

print('complete 1')

# Load pipeline config and build a detection model
PATH_TO_CFG = 'C:\\Users\\lucam\\Documents\\Tensorflow\\data\\models\\ssd_resnet101_v1_fpn_640x640_coco17_tpu-8\\pipeline.config'
print(PATH_TO_CFG)
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

print('complete 2')

# Restore checkpoint
PATH_TO_CKPT = 'C:\\Users\\lucam\\Documents\\Tensorflow\\data\\models\\ssd_resnet101_v1_fpn_640x640_coco17_tpu-8\\checkpoint\\'
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

print('complete 3')

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

PATH_TO_LABELS = 'C:\\Users\\lucam\\Documents\\Tensorflow\\data\\models\\ssd_resnet101_v1_fpn_640x640_coco17_tpu-8\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

print('complete 4')
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.65,
          agnostic_mode=False)

    cv2.imshow('no detection', cv2.resize(image_np, (800, 600)))
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    print('passed imshow')

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()