import cv2
import imutils

class AspectAwarePreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        # initialize the difference of width and height along with grabbing height and width of input image
        d_W, d_H = 0, 0
        h, w = image.shape[:2]

        # resize based on the smaller dimension of the input 
        if w < h:
            image = imutils.resize(image, width=self.width)
            d_H = (image.shape[0] - self.height) // 2
        else:
            image = imutils.resize(image, height=self.height)
            d_W = (image.shape[1] - self.width) // 2
        
        # crop the image and resize ot desired dimension
        h, w = image.shape[:2]
        image = image[d_H:h - d_H, d_W:w - d_W]

        return cv2.resize(image, (self.width, self.height))