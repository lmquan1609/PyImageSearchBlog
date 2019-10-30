# USAGE
# python train_clr.py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import CyclicLR
from config import cifar10_config as config
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# load the training and testing dataset, scale data
print('[INFO] loading CIFAR-10 data...')
(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_X, test_X = map(lambda X: X.astype('float'), [train_X, test_X])
train_mean = train_X.mean(axis=0)
train_X, test_X = map(lambda X: X-train_mean, [train_X, test_X])

# binarize the labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1, zoom_range=0.2, horizontal_flip=True)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=config.MIN_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=len(lb.classes_))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# initialize the cyclical lr callback
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