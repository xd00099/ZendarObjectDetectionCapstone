

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # plt.imshow(img)
    # plt.show()



def run_yolo(img, cfg_path, class_path, weights_path):

    # Model Setup
    config_path = cfg_path
    classes_path = class_path
    weights_path = weights_path

    # Classes Path
    classes = None
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # Config
    net = cv2.dnn.readNet(weights_path, config_path)


    # Results
    labeled_images = []
    image_boxes_np = []
    box_coordinates = []

    # Pass Image
    image = img
    # plt.imshow(image)
    # plt.show()

    # Blob
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    output_labels = []
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        # Remove Classes that are not
        if class_ids[i] == 0 or class_ids[i] == 2 or class_ids[i] == 5 or class_ids[i] == 7:
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h), classes, COLORS)
            output_labels.append(classes[class_ids[i]])
            box_coordinates.append(box)

    image_boxes_np.append(box_coordinates)
    labeled_images.append(image)


    image_boxes_np = np.array(image_boxes_np, dtype=object)
    return labeled_images[0], list(zip(image_boxes_np[0], output_labels))

