# USAGE
# python train.py --dataset data --model output/activity.model --label-bin output/lb.pickle --epochs 50
import matplotlib
matplotlib.use('Agg')

from keras.applications import ResNet50
from pyimagesearch.nn.conv import FCHeadNet
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocesing import AspectAwarePreprocessor, ColorChannelSwitchPreprocessor
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
ap.add_argument('-m', '--model', required=True, help='Path to serialized model')
ap.add_argument('-l', '--label-bin', required=True, help='Path to label binarizer')
ap.add_argument('-e', '--epochs', type=int, default=25, help='# of epochs to train the network')
ap.add_argument('-p', '--plot', type=str, default='plot.png', help='Path to output plot.accuracy plot')
args = vars(ap.parse_args())

# grab the list of image paths in dataset
image_paths = list(paths.list_images(args['dataset']))

# initialize preprocessors
ccsp = ColorChannelSwitchPreprocessor(cv2.COLOR_BGR2RGB)
aap = AspectAwarePreprocessor(224, 224)

# initialize the set of labels classified during training
LABELS = set(['weight_lifting', 'tennis', 'football'])

# initialize Dataset Loader and load dataset
sdl = SimpleDatasetLoader(preprocessors=[ccsp, aap], labels_set=LABELS)
data, labels = sdl.load(image_paths, verbose=100)

# Binarze the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# split dataset into ratio (75, 25)
train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# initialize the training, validation data augmentation
train_aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)
val_aug = ImageDataGenerator()

# define mean of Imagenet and set the mean substraction value for each of the data augmentaiton
mean = np.array([123.68, 116.779, 103.939], dtype='float32')
train_aug.mean = mean
val_aug.mean = mean

# load the ResNet50 model with cutting top FC layers
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# define head model to replace the cut FC layers of ResNet50
head_model = FCHeadNet.build(base_model, 512, len(lb.classes_))

# combine these 2 model into one
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze the base model layers for fine-tuning
for layer in base_model.layers:
    layer.trainable=False

# compile the model
print('[INFO] Compiling the model...')
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4/args['epochs'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network for warming-up
print('[INFO] warming-up the network...')
H = model.fit_generator(train_aug.flow(train_X, train_y, batch_size=32), validation_data=val_aug.flow(test_X, test_y), validation_steps=len(test_X)//32, steps_per_epoch=len(train_X)//32, verbose=2, epochs=args['epochs'])

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(test_X, batch_size=32)
print(classification_report(test_y.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
plt.style.use('ggplot')
N = list(range(args['epochs']))
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['acc'], label='train_acc')
plt.plot(N, H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

# serialize the model to disk
print('[INFO] serialzing network...')
model.save(args['model'])

# seriali the label binarizer to disk
f = open(args['label_bin'], 'wb')
f.write(pickle.dumps(lb))
f.close()