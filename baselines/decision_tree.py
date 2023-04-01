# import the necessary packages
import numpy as np
import cv2
import os

# import the necessary packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV


from imutils import paths
import argparse


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA, kernel_width=100, kernel_height=100, kernel_conv=np.ones((7, 7), np.float32)/25):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_conv = kernel_conv

    def preprocess(self, image):
        image = cv2.resize(image, (self.width, self.height),
                           interpolation=self.inter)
        # image = cv2.blur(image, (self.kernel_width, self.kernel_height))
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return image


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
trainImagePaths = list(paths.list_images(args["dataset"] + '/train'))
testImagePaths = list(paths.list_images(args["dataset"] + '/test'))

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])


print("[INFO] Starting load train dataset...")
(train_data, train_labels) = sdl.load(trainImagePaths, verbose=500)
train_data = train_data.reshape((train_data.shape[0], 3072))
print("[INFO] Loaded train dataset...")

print("[INFO] Starting load test dataset...")
(test_data, test_labels) = sdl.load(testImagePaths, verbose=500)
test_data = test_data.reshape((train_data.shape[0], 3072))
print("[INFO] Loaded test dataset...")

# show some information on memory consumption of the images
print("[INFO] train features matrix: {:.1f}MB".format(
    train_data.nbytes / (1024 * 1000.0)))
print("[INFO] test features matrix: {:.1f}MB".format(
    test_data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
trle = LabelEncoder()
train_labels = trle.fit_transform(train_labels)

tele = LabelEncoder()
test_labels = tele.fit_transform(test_labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, validX, trainY, validY) = train_test_split(
    train_data, train_labels, test_size=0.25, random_state=42)

# defining parameter range
param_grid = [
  {"C":np.logspace(-1,1,7), "penalty":["l1","l2"]},
 ]


print("[INFO] evaluating Logistic classifier...")
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid_search.fit(trainX, trainY)

accuracy = grid_search.best_score_ * 100
print("[INFO] accuracy for our training dataset with tuning is : {:.2f}%".format(
    accuracy))

print("[INFO] best parameter after tuning...")
# print best parameter after tuning
print(grid_search.best_params_)

# print classification report
print(classification_report(test_labels, grid_search.best_estimator_.predict(
    test_data), target_names=tele.classes_))
