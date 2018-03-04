# /media/sf_XubuntuShareFile/frozen_inference_graph.pb
# /media/sf_XubuntuShareFile/frozen_inference_graph_resnet.pb
# /media/sf_XubuntuShareFile/SSD_Mobilenet.pb
#                            ssd_inception_v2
#                            faster_rcnn_inception_v2
#                            faster_rcnn_resnet50
#                            faster_rcnn_resnet50_lowproposals
#                            faster_rcnn_resnet101
#                            rfcn_resnet101
#                            faster_rcnn_resnet101_lowproposals
#                            inception_resnet_v2_atrous
#                            inception_resnet_v2_atrous_lowproposals
#                            faster_rcnn_nas
#                            faster_rcnn_nas_lowproposals
# Person_SSD_Mobilenet_Xincheng_BLDG1_81559.pb

import tensorflow as tf
import cv2
import numpy as np
import time

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('/media/sf_XubuntuShareFile/faster_rcnn_nas_lowproposals.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
image_np = cv2.imread('/media/sf_XubuntuShareFile/street_1.jpg')
image_np_expanded = np.expand_dims(image_np, axis=0)

sess = tf.Session(graph=detection_graph)

start = time.time()
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})
end = time.time()
print('Running time: %.2fs.' % (end-start))


boxes = boxes[0]
scores = scores[0]
classes = classes[0]
high_conf_indices = [i for i,v in enumerate(scores) if v > 0.5]
high_conf_boxes = boxes[high_conf_indices]
high_conf_classes = classes[high_conf_indices]


def show_detection(image_np, boxes):
    cv2.imshow('Detection', image_np)
    (im_height, im_width) = image_np.shape[:2]

    num_boxes = boxes.shape[0]

    for cls_ind in range(num_boxes):    
        cls = int(high_conf_classes[cls_ind])
#        cls = high_conf_classes[cls_ind]

        ymin = int(boxes[cls_ind][0] * im_height)
        xmin = int(boxes[cls_ind][1] * im_width)
        ymax = int(boxes[cls_ind][2] * im_height)
        xmax = int(boxes[cls_ind][3] * im_width)
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
        cv2.putText(image_np,str(cls),(xmax,ymax),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imshow('Detection', image_np)
    cv2.waitKey(0)


#    for box in boxes:
#        ymin = int(box[0] * im_height)
#        xmin = int(box[1] * im_width)
#        ymax = int(box[2] * im_height)
#        xmax = int(box[3] * im_width)
#        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
#        cv2.putText(image_np,'1',(xmax,ymax),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),2)
#        cv2.imshow('Detection', image_np)
#    cv2.waitKey(0)

show_detection(image_np,high_conf_boxes)

