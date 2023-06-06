import cv2


class ImageRecognitionTrain:
    def __init__(self, image, kernel_size):
        self.image = image
        self.kernel_size = kernel_size

    def preprocessing(self, width, height):
        preprocess = cv2.resize(self.image, (width, height))
        preprocess = cv2.cvtColor(preprocess, cv2.COLOR_BGR2GRAY)
        preprocess = cv2.GaussianBlur(
            preprocess, (self.kernel_size, self.kernel_size), 0
        )

        return preprocess