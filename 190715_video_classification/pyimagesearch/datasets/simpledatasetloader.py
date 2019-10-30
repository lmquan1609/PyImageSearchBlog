import cv2
import os
import numpy as np

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None, labels_set=None):
        if preprocessors is None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors
        
        if labels_set is None:
            self.labels_set = set([])
        else:
            self.labels_set = labels_set

    def load(self, image_paths, verbose=50):
        data, labels = [], []
        count = 0
        for i, image_path in enumerate(image_paths):
            # split the label from its path
            label = image_path.split(os.path.sep)[-2]

            # check whether label is in target labels set
            if label not in self.labels_set:
                continue
            
            count += 1

            if verbose and count % verbose == 0:
                print(f'[INFO] loading {count} images...')

            # load the image, preprcess it
            image = cv2.imread(image_path)
            for p in self.preprocessors:
                image = p.preprocess(image)

            data.append(image)
            labels.append(label)

        return np.array(data), np.array(labels)