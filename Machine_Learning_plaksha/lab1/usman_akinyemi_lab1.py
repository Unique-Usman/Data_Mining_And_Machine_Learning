"""
A Python script to extract features from picture of a chromosomes
"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def readImg(filename):
    """
    Read image from a file and convert image to RGB

    Args:
        filename: file to be read
    """
    chromosomes = cv2.imread(filename)
    chromosomes = cv2.cvtColor(chromosomes, cv2.COLOR_BGR2RGB)
    return chromosomes


def plotImage(imgread):
    """
    Plot an image using plt

    Args:
        imgread: image read by readImg and converted to RGB
    """

    plt.imshow(imgread)
    plt.axis('off')
    plt.show()


def convertImageToGray(img):
    """
    Convert image to gray
    Args:
        img: image read by readImg and converted to RGB
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Image with in Gray form", img)
    cv2.waitKey(0)
    return img_gray


def applyMorpholorgical(img):
    """
    Apply morphological opening for background removal

    Args:
        img: Gray image
    """
    kernelSize = (5, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening: ({}, {})".format(
        kernelSize[0], kernelSize[1]), opening)
    cv2.waitKey(0)
    return opening


def imgThreshold(img):
    """
    Thresholding the image

    Args:
        img: image that in which morphological has been applied.
    """
    _, binary_image = cv2.threshold(
        img, 0, 255,  cv2.THRESH_OTSU)
    cv2.imshow("Image after threshold", binary_image)
    cv2.waitKey(0)
    return binary_image


def findContours(img):
    """
    Finding contours of the image

    Args:
        img: Image with already applied thresholding
    """
    contours, hierarchy = cv2.findContours(
        img, 1, 2)
    return contours, hierarchy


def getImageFeatures(contours, img):
    """
    Get the image features and also draw a boundind box for each
    chrmosomes
    Args:
        contours: Countours from the previous function
        img: Image of the chromosomes.
    """
    features = []

    # recommened threshold value
    threshold_value = 5

    if len(contours) < threshold_value:
        return
    # setting minimum box area to avoid overlap and filter small box
    min_contour_area = 500
    for i, contour in enumerate(contours):
        Area = cv2.contourArea(contour)
        if Area > min_contour_area:
            perimeter = cv2.arcLength(contour, True)
            Circularity = (4 * np.pi * Area) / (perimeter ** 2)
            Perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            height = h
            width = w
            features.append({
                'Area': Area,
                'Perimeter': Perimeter,
                'Circularity': Circularity,
                'Height': height,
                'Width': width,
            })
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    chromosomes_features = pd.DataFrame(features)
    return chromosomes_features


def standard_normalization(features):
    """
    Perform the standadization and normalization of the features

    Args:
        features: The features of the images
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # # Separate features from the DataFrame
    features = features[[
        'Area', 'Perimeter', 'Circularity', 'Height', 'Width']]

    # Standardization
    scaler_standard = StandardScaler()
    features_standardized = pd.DataFrame(
        scaler_standard.fit_transform(features), columns=features.columns)

    # Normalization
    scaler_minmax = MinMaxScaler()
    features_normalized = pd.DataFrame(
        scaler_minmax.fit_transform(features), columns=features.columns)

    # Display the standardized and normalized features
    print("Standardized Features:")
    print(features_standardized.std())
    print(features_standardized)
    print(features_standardized.mean())

    print("\nNormalized Features:")
    print(features_normalized)


if __name__ == "__main__":
    chromosomes = readImg("chromosomes.jpg")
    plotImage(chromosomes)
    img_gray = convertImageToGray(chromosomes)
    openinng = applyMorpholorgical(img_gray)
    img = imgThreshold(openinng)
    contours, hierarchy = findContours(img)
    features = getImageFeatures(contours, chromosomes)
    print(features)
    standard_normalization(features)
