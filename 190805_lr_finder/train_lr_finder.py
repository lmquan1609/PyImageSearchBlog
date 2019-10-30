# USAGE
# python train_lr_finder.py --lr-find 1
# python train_lr_finder.py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.callbacks import LearningRateFinder
from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import CyclicLR
from config import fashion_mnnist_config as config
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--lr-find', type=int, default=0, help='whether or not to find optimal lr')
args = vars(ap.parse_args())

# load the training and testing dataset
print('[INFO] loading Fashion data...')
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

# Fashion MNIST images are 28 x 28 but the network is applicable to 32 x 32 images
train_X, test_X = map(lambda X: np.array([cv2.resize(x, (32, 32)) for x in X]), [train_X, test_X])

# scale data
train_X, test_X = map(lambda X: X.astype('float')/255., [train_X, test_X])

# reshape the data matrices to include a channel dim
train_X, test_X = map(lambda X: X.reshape((len(X), 32, 32, 1)), [train_X, test_X])

# binarize the labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1, zoom_range=0.2, horizontal_flip=True)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=config.MIN_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=1, classes=len(lb.classes_))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# check to see if we are trying to find an optimal lr before training for the full number of epochs
if args['lr_find'] > 0:
    # initialize the lr finder and then train with lr ranging from 1e-10 to 1e+1
    print('[INFO] finding lr...')
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(train_X, train_y, batch_size=config.BATCH_SIZE), 1e-10, 1e1, steps_per_epoch=np.ceil((len(train_X) / float(config.BATCH_SIZE))), batch_size=config.BATCH_SIZE)

    # plot the loss for the various lr and save the resulting plot to disk
    lrf.plot_loss()
    plt.savefig(config.LRFINND_PLOT_PATH)

    # exit the script so we can adjust our learning rates in the config and then train the network for our full set of epochs
    print('[INFO] learning rate finder complete')
    print('[INFO] examine plot and adjust learning rates before training')
    sys.exit(0)

# otherwise, we have already defined a learning rate space to train over, so compute the step size and initialize the cyclic learning rate method
print(f'[INFO] using {config.CLR_METHOD} method')
clr = CyclicLR(base_lr=config.MIN_LR, max_lr=config.MAX_LR, step_size=config.STEP_SIZE * (train_X.shape[0] // config.BATCH_SIZE), mode=config.CLR_METHOD)

# train the network
print('[INFO] training network...')
H = model.fit_generator(aug.flow(train_X, train_y, batch_size=config.BATCH_SIZE), validation_data=(test_X, test_y), steps_per_epoch=len(train_X)//config.BATCH_SIZE, epochs=config.NUM_EPOCHS, verbose=2, callbacks=[clr])

# evaluate the network and show a classification report
print('[INFO] Evaluating network...')
preds = model.predict(test_X, batch_size=config.BATCH_SIZE)
print(classification_report(test_y.argmax(axis=1), preds.argmax(axis=1), target_names=config.CLASSES))

# construct a plot that plots and saves the training history
N = list(range(config.NUM_EPOCHS))
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(config.TRAINING_PLOT_PATH)

# plot the lr history
N = list(range(len(clr.H['lr'])))
plt.figure()
plt.plot(N, clr.H['lr'])
plt.title('Cyclical LR')
plt.xlabel('Training Iterations')
plt.ylabel('lr')
plt.legend(loc='lower left')
plt.savefig(config.CLR_PLOT_PATH)