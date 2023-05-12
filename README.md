#Color Correction Using Histogram Processing

The code performs the following steps:

1. Loads the image specified by img_path.
2. Displays the original image.
3. Converts the image to grayscale.
4. Computes and plots the histogram and cumulative distribution of pixel intensities.
5. Performs histogram equalization on the grayscale image.
6. Displays the corrected grayscale image.
7. Performs histogram equalization and intensity transformation on the individual color channels (red, green, and blue).
8. Displays the actual and target cumulative distribution functions (CDFs) for each color channel.
9. Shows the corrected image in each color channel.
10. Displays the original image and the transformed image with corrected color channels.


Installation

To run the code, you need to have the following libraries installed:
- NumPy
- Matplotlib
- scikit-image
- scipy


1. NumPy

-pip install numpy

2. Matplot.lib

-pip install matplotlib

3. Scikit-image

-pip install scikit-image

4. SciPy

-pip install scipy

Note: Use pip3 command if you have macOS computer

Notes
- Make sure to provide the correct path to the image you want to process.
- The code assumes that the image is in RGB format. If your image is in a different format, you may need to modify the code accordingly.
- The code uses predefined target bins and frequencies for histogram equalization. You can adjust these values based on your specific requirements.
- The code includes examples of histogram equalization using both Cauchy and Logistic distributions. You can explore other distributions by modifying the code accordingly.

