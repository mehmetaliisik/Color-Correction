import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import histogram, cumulative_distribution
from scipy.stats import cauchy, logistic

# Load the image
img = imread('13.jpg')

# Display the original image
plt.imshow(img)
plt.title('Maltepe Beach')

# Create subplots for grayscale image and histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Convert the image to grayscale
img_gray = rgb2gray(img)

# Display the grayscale image
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title('Grayscale Image')

# Compute and plot the histogram and cumulative distribution of pixel intensities
ax1 = ax[1]
ax2 = ax1.twinx()
freq_h, bins_h = histogram(img_gray)
freq_c, bins_c = cumulative_distribution(img_gray)
ax1.step(bins_h, freq_h*1.0/freq_h.sum(), c='b', label='PDF')
ax2.step(bins_c, freq_c, c='r',  label='CDF')
ax1.set_ylabel('PDF', color='b')
ax2.set_ylabel('CDF', color='r')
ax[1].set_xlabel('Intensity value')
ax[1].set_title('Histogram of Pixel Intensity')

# Convert the grayscale image to 8-bit representation
image_intensity = img_as_ubyte(img_gray)

# Compute the cumulative distribution of the intensity values
freq, bins = cumulative_distribution(image_intensity)

# Define the target bins and frequencies for histogram equalization
target_bins = np.arange(255)
target_freq = np.linspace(0, 1, len(target_bins))

# Interpolate the actual CDF to match the target CDF
new_vals = np.interp(freq, target_freq, target_bins)

# Create subplots for histogram equalized image and transformed image
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Plot the actual and target CDFs
ax[0].step(bins, freq, c='b', label='Actual CDF')
ax[0].plot(target_bins, target_freq, c='r', label='Target CDF')
ax[0].legend()
ax[0].set_title('Grayscale: Actual vs. Target Cumulative Distribution')

# Display the corrected image after histogram equalization
ax[1].imshow(new_vals[image_intensity].astype(np.uint8), cmap='gray')
ax[1].set_title('Corrected Image in Grayscale')

# Function to show the CDFs for individual color channels
def show_linear_cdf(image, channel, name, ax):
    # Convert the channel to 8-bit representation
    image_intensity = img_as_ubyte(image[:, :, channel])
    
    # Compute the cumulative distribution of the channel intensities
    freq, bins = cumulative_distribution(image_intensity)
    
    # Plot the actual and target CDFs
    ax.step(bins, freq, c='b', label='Actual CDF')
    ax.plot(target_bins, target_freq, c='r', label='Target CDF')
    ax.legend()
    ax.set_title('{} Channel: Actual vs. Target Cumulative Distribution'.format(name))

# Function to perform histogram equalization and intensity transformation on an individual channel
def linear_distribution(image, channel):
    image_intensity = img_as_ubyte(image[:,:,channel])  # Convert the specified channel to 8-bit representation
    freq, bins = cumulative_distribution(image_intensity)  # Compute the cumulative distribution of the channel intensities
    target_bins = np.arange(255)  # Define the target bins for histogram equalization
    target_freq = np.linspace(0, 1, len(target_bins))  # Define the target frequencies for histogram equalization
    new_vals = np.interp(freq, target_freq, target_bins)  # Interpolate the actual CDF to match the target CDF
    return new_vals[image_intensity].astype(np.uint8)  # Apply the intensity transformation to the channel

fig, ax = plt.subplots(3,2, figsize=(12,14))  # Create subplots for displaying the results

# Apply histogram equalization and intensity transformation to the red, green, and blue channels
red_channel = linear_distribution(img, 0)
green_channel = linear_distribution(img, 1)
blue_channel = linear_distribution(img, 2)

show_linear_cdf(img, 0, 'Red', ax[0,0])  # Show the CDF for the red channel
ax[0,1].imshow(red_channel, cmap='Reds')  # Display the corrected image in the red channel
ax[0,1].set_title('Corrected Image in Red Channel')

show_linear_cdf(img, 1, 'Green', ax[1,0])  # Show the CDF for the green channel
ax[1,1].imshow(green_channel, cmap='Greens')  # Display the corrected image in the green channel
ax[1,1].set_title('Corrected Image in Green Channel')

show_linear_cdf(img, 2, 'Blue', ax[2,0])  # Show the CDF for the blue channel
ax[2,1].imshow(blue_channel, cmap='Blues')  # Display the corrected image in the blue channel
ax[2,1].set_title('Corrected Image in Blue Channel')

fig, ax = plt.subplots(1,2, figsize=(15,5))  # Create subplots for displaying the original and transformed images

ax[0].imshow(img)  # Display the original image
ax[0].set_title('Original Image')

# Display the transformed image by stacking the corrected red, green, and blue channels
ax[1].imshow(np.dstack([red_channel, green_channel, blue_channel]))
ax[1].set_title('Transformed Image')

# Saving the transformed image to a jpg file
cv2.imwrite("transformed.jpg", np.dstack([red_channel, green_channel, blue_channel])) 

def individual_channel(image, dist, channel):
    im_channel = img_as_ubyte(image[:,:,channel])  # Convert the specified channel to 8-bit representation
    freq, bins = cumulative_distribution(im_channel)  # Compute the cumulative distribution of the channel intensities
    new_vals = np.interp(freq, dist.cdf(np.arange(0,256)), np.arange(0,256))  # Interpolate the channel's CDF based on the provided distribution
    return new_vals[im_channel].astype(np.uint8)  # Apply the intensity transformation to the channel

def distribution(image, function, mean, std):
    dist = function(mean, std)  # Create a distribution based on the provided parameters
    fig, ax = plt.subplots(1,2, figsize=(15,5))  # Create subplots for displaying the results

    image_intensity = img_as_ubyte(rgb2gray(image))  # Convert the image to grayscale
    freq, bins = cumulative_distribution(image_intensity)  # Compute the cumulative distribution of the intensity values
    ax[0].step(bins, freq, c='b', label='Actual CDF')  # Plot the actual CDF
    ax[0].plot(dist.cdf(np.arange(0,256)), c='r', label='Target CDF')  # Plot the target CDF based on the provided distribution
    ax[0].legend()
    ax[0].set_title('Actual vs. Target Cumulative Distribution')

    # Apply individual channel intensity transformation based on the provided distribution
    red = individual_channel(image, dist, 0)
    green = individual_channel(image, dist, 1)
    blue = individual_channel(image, dist, 2)

    ax[1].imshow(np.dstack((red, green, blue)))  # Display the transformed image
    ax[1].set_title('Transformed Image')
    return ax

# Apply the distribution function with Cauchy distribution and parameters (mean=90, std=30)
distribution(img, cauchy, 90, 30)

# Apply the distribution function with Logistic distribution and parameters (mean=90, std=30)
distribution(img, logistic, 90, 30)
