# USAGE
# python predict_video.py --model output/activity.model --label-bin output/lb.pickle --input example_clips/tennis.mp4 --output output/tennis_1frame.avi --size 1
from keras.models import load_model
from pyimagesearch.preprocesing import *
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to serialized model')
ap.add_argument('-l', '--label-bin', required=True, help='Path to label binarizer')
ap.add_argument('-i', '--input', required=True, help='Path to input video')
ap.add_argument('-o', '--output', required=True, help='Path to output video')
ap.add_argument('-s', '--size', type=int, default=128, help='Size of queue for averaging')
args = vars(ap.parse_args())

# load the model and its label binarizer
print('[INFO] loading model and label binarizer...')
model = load_model(args['model'])
lb = pickle.loads(open(args['label_bin'], 'rb').read())

# initialize preprocessors
ccsp = ColorChannelSwitchPreprocessor(cv2.COLOR_BGR2RGB)
aap= AspectAwarePreprocessor(224, 224)

# initialize the image mean for mean substraction along with preds queu
mean = np.array([123.68, 116.779, 103.939], dtype='float32')
Q = deque(maxlen=args['size'])

# initialize the video stream, pointer to output video file, and frame dimensions
camera = cv2.VideoCapture(args['input'])
writer = None
W, H = None, None

while True:
    # read frame from the file
    grabbed, frame = camera.read()

    if not grabbed:
        break

    if W is None or H is None:
        H, W = frame.shape[:2]

    # clone the current frame, and preprocess it
    output = frame.copy()
    frame = aap.preprocess(ccsp.preprocess(frame)).astype('float32')
    frame -= mean

    # predict on the frame and then update the predictions queue
    preds = model.predict(np.expand_dims(frame,axis=0))[0]
    Q.append(preds)

    # perform prediction averaging over the current history of prev preds
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]

    # draw the activity on the output frame
    text = "activity: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

    # check if the video writer is None
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # write the output frame to disk
    writer.write(output)

    # show the output image
	# cv2.imshow("Output", output)
	# key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	#	break

# release the file pointers
print('[INFO] cleaning up...')
writer.release()
camera.release()