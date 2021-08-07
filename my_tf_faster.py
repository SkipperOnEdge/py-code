import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Load pipeline config and build a detection model
PATH_TO_CFG = 'C:\\Users\\lucam\\Documents\\Tensorflow\\data\\models\\ssd_mobilenet_v2_320x320_coco17_tpu-8\\pipeline.config'
print(PATH_TO_CFG)
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
PATH_TO_CKPT = 'C:\\Users\\lucam\\Documents\\Tensorflow\\data\\models\\ssd_mobilenet_v2_320x320_coco17_tpu-8\\checkpoint\\'
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

PATH_TO_LABELS = 'C:\\Users\\lucam\\Documents\\Tensorflow\\data\\models\\ssd_mobilenet_v2_320x320_coco17_tpu-8\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

print('complete 4')
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    image_np_expanded = np.expand_dims(image_np, axis=0)
    height, width, channel = image_np.shape
    #print('height and width:')
    #print(height)
    #print(width)

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
    y = (detections['detection_boxes'][0][0][0].numpy() + detections['detection_boxes'][0][0][2].numpy())* 480 / 2
    x = (detections['detection_boxes'][0][0][1].numpy() + detections['detection_boxes'][0][0][3].numpy())* 640 / 2
    tx = 640/2
    ty = 480/2
    cv2.circle(image_np_with_detections,(int(x), int(y)), 10, (0,0,255), -1 )
    cv2.circle(image_np_with_detections,(int(tx), int(ty)), 10, (0,0,255), -1 )
    print('x offset: ' + str(tx - x))
    print('y offset: ' + str(ty - y))

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

#print(detections['detection_boxes'][0].numpy())
#print(detections['detection_scores'][0].numpy())
#print(detections['detection_classes'][0].numpy())
cap.release()
cv2.destroyAllWindows()