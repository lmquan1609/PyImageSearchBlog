import cv2

class ColorChannelSwitchPreprocessor:
    def __init__(self, switch_method=cv2.COLOR_BGR2GRAY):
        self.switch_method = switch_method
    def preprocess(self, image):
        return cv2.cvtColor(image, self.switch_method)