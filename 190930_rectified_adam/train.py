# USAGE
# python train.py --plot cifar10_adam.png --optimizer adam
# python train.py --plot cifar10_radam.png --optimizer radam
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import ResNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras_radam import RAdam
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--plot', type=str, required=True, help='Path to output training plot')
ap.add_argument('-o', '--optimizer', type=str, default='adam', choices=['adam', 'radam'], help='Type of optimizer')
args = vars(ap.parse_args())

# initialize the number of epochs to train and batch size
EPOCHS = 75
BATCH_SIZE = 128

# load the training and testing data, scale it to range [0, 1]
print('[INFO] loading CIFAR-10 data...')
(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_X, test_X = map(lambda X: X.astype('float')/255., [train_X, test_X])

# binarize the label
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# initialize the label names for the CIFAR-10
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# check if we are using Adam
if args['optimizer'] == 'adam':
    # initialize the Adam optimizer
    print('[INFO] using Adam optimizer')
    opt = Adam(lr=1e-3)
else:
    # initialize the RAdam optimizer
    print('[INFO] using Rectified Adam optimizer')
    opt = RAdam(learning_rate=1e-3, total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)

# initialize optimizer and model, then compile it
model = ResNet.build(32, 32, 3, len(lb.classes_), (9, 9, 9), (64, 64, 128, 256), reg=5e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
H = model.fit_generator(aug.flow(train_X, train_y, batch_size=BATCH_SIZE), steps_per_epoch=len(train_y)//BATCH_SIZE, validation_data=(test_X, test_y), epochs=EPOCHS, verbose=2)

# evaluate the network
print('[INFO] evaluating network...')
preds = model.predict(test_X, batch_size=BATCH_SIZE)
print(classification_report(test_y.argmax(axis=1), preds.argmax(axis=1), target_names=label_names))

# determine the number of epochs and then construct the plot title
N = np.arange(0, EPOCHS)
title = "Training Loss and Accuracy on CIFAR-10 ({})".format(
	args["optimizer"])
 
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title(title)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])