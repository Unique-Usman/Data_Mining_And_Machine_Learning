"""
# A python script that apply principal component analysis
on the given image. Using numpy to calculate the PCA
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import cv2


def readImg(filename):
    """
    Read image from a file and convert the image
    to GrayScale and also convert it to Double

    Args:
        filename: file to be read
    """
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img_grayfloat64 = img_gray.astype(np.float64)
    return img_grayfloat64


def subtract_mean(img_grayfloat64):
    """
    Compute the mean of each column and subtract it
    from the image

    Args:
        img_grayfloat64: the image in grayscale and also in double
    """
    img_grayfloat64_mean = np.mean(img_grayfloat64, axis=0)
    standardized_img = img_grayfloat64 - img_grayfloat64_mean
    return standardized_img, img_grayfloat64_mean


def find_eignvalues_and_vector(image_mean_subtracted, mean_column, filename):
    """
    Compute the covariance matrix 
    Get the eigen_vectors by eigen_value
    Sort the eigen_vectors and eigen_value 
    in decending order.
    Use different number of Principal components and then reconstruct the image

    Args:
        image_mean_subtracted = The standardized_img
        mean_column: The mean of the column. It is needed to regenerate the image
        filename: To plot the original image and the grayscale image
    """
    image_covariance = np.cov(image_mean_subtracted.T)
    img_eigenvalues, img_eigenvectors = np.linalg.eig(image_covariance)
    sorted_indices = np.argsort(img_eigenvalues)
    sorted_indices = sorted_indices[::-1]

    sorted_eigenvectors = img_eigenvectors[:, sorted_indices]

    Num_components = [10, 20, 30, 40, 50, 60, 91, 300]

    output_images = []

    for num in Num_components:
        selected_components = sorted_eigenvectors[:, :num]

        projected_data = np.dot(selected_components.T,
                                image_mean_subtracted.T).T
        reconstructed_image = np.dot(
            selected_components, projected_data.T).T + mean_column
        output_images.append(reconstructed_image)

    # Calculate the number of rows and columns for subplots
    num_rows = (len(output_images) + 2) // 4 + 1  # Adding 2 for the original grayscale and original RGB image
    num_cols = min(len(output_images), 4)

    # First, display the grayscale images generated from different numbers of components
    plt.figure(figsize=(4 * num_cols, 6 * num_rows))
    plt.suptitle("Dimensionality Reduction using PCA")
    for i, img in enumerate(output_images):
        img = img.astype(np.uint8)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{Num_components[i]} Components")
        plt.axis('off')

    # Then, display the original grayscale image
    original_img_gray = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)
    plt.subplot(num_rows, num_cols, len(output_images) + 1)
    plt.imshow(original_img_gray, cmap='gray')
    plt.title("GrayScale image(Satellite view of plaksha university)")
    plt.axis('off')

    # Finally, display the original RGB image
    original_img_rgb = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    plt.subplot(num_rows, num_cols, len(output_images) + 2)
    plt.imshow(original_img_rgb)
    plt.title("Original RGB Image")
    plt.axis('off')
    plt.xlim(0, original_img_rgb.shape[1])
    plt.ylim(original_img_rgb.shape[0], 0)
    plt.axis('on')
    plt.show() 


def check_dim_for_95_variance(img_grayfloat64):
    """
    It shows how the dim 91 explain the 95% variance in
    the data.

    Args:
        img_grayfloat64: The image in a gray scale and converted to double
    """
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(img_grayfloat64)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components needed for 95% variance explained
    num_components_95_variance = np.argmax(cumulative_variance >= 0.95) + 1

    if num_components_95_variance == 91:
        print("""It is true that the 91 dimension is able to explain the 95% of the 
              variability in the data.""")

 
    
if __name__ == "__main__":
    img_grayfloat64 = readImg("./sat_image_plaksha.jpg")
    image_mean_subtracted, img_grayfloat64_mean = subtract_mean(
        img_grayfloat64)
    find_eignvalues_and_vector(image_mean_subtracted, img_grayfloat64_mean, "./sat_image_plaksha.jpg")
    check_dim_for_95_variance(img_grayfloat64)
