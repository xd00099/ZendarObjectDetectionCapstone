

import cv2
import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


## Data Loader

# Path:
mypath = '/Users/consti/Desktop/data/'

# Load Files
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Filter For .npz Files
onlyfiles = [item for item in onlyfiles if '.npz' in item]
print('Number of Files: ',len(onlyfiles))

# Extract Images
images = []


# Iterate over Files:
for files in onlyfiles:
    image_file = np.load(mypath + files, allow_pickle = True)
    lst = image_file.files
    images.extend(list(image_file[lst[1]].item().values()))


## Model Setup

ap = argparse.ArgumentParser()

#ap.add_argument('-i', '--image', required=True,help = 'path to input image')

#ap.add_argument('-c', '--config', required=True,help = 'path to yolo config file')
config_path = '/Users/consti/Documents/Berkeley/Capstone/scripts/object-detection-opencv-master/yolov3.cfg'

#ap.add_argument('-w', '--weights', required=True,help = 'path to yolo pre-trained weights')
weights_path = '/Users/consti/Documents/Berkeley/Capstone/scripts/object-detection-opencv-master/yolov3.weights'

#ap.add_argument('-cl', '--classes', required=True,help = 'path to text file containing class names')
classes_path = '/Users/consti/Documents/Berkeley/Capstone/scripts/object-detection-opencv-master/yolov3.txt'

args = ap.parse_args()



def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
#image = cv2.imread(args.image)
#image = cv2.imread(images[0])
image = images[0]

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

## Labels
classes = None
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weights_path, config_path)
#net = cv2.dnn.readNet(args.weights, args.config)

labeled_images = []

for idx_img, files in enumerate(onlyfiles):

    image = images[idx_img]

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

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
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    labeled_images.append(image)
    #cv2.imshow("object detection", image)
    #cv2.waitKey()


idx = 0
for files in onlyfiles:
    cv2.imwrite('/Users/consti/Documents/Berkeley/Capstone/labeled_data/'+files+'_object-detection.jpg', labeled_images[idx])
    cv2.destroyAllWindows()
    idx += 1
