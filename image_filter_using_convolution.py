'''
A convolution kernel is a 2D matrix used to filter images
*Typically MxN matrix where both M and N are odd
How to use kernels to sharpen/blur images
- assume center of kernel is positioned over a specific pixel
- multiply the value of each element in the kernel with the corresponding pixel element
- sum the result of the multiplication and compute average
- replace the value of pixel with the average value computed

IDENTITY KERNEL:
[000]
[010]
[000]
Multiplying it with any other matrix returns original matrix
'''
import cv2
import numpy as np
'''
FOLLOWING STEPS:
- read test image
- define identity kernel
- use filter2D() to perform linear filtering
- Display orignal and filtered images

filter2D(src, ddepth, kernel)
src = source image, ddepth = depth of resulting image, -1 = same depth as orig, kernel = kernel applied to image
'''
image = cv2.imread("image_filter.webp")
#apply identity kernel
kernel1 = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,0]])
identity = cv2.filter2D(src = image, ddepth = 1-1, kernel = kernel1)
cv2.imshow("Orignal", image)
cv2.imshow("identity", identity)
cv2.waitKey()
cv2.destroyAllWindows()

#Blurring image using convultion kernel
#*DIVIDE KERNEL BY SUM OF ALL ELEMENTS TO NORMAL, MUST BE BTWN [0,1]
kernel2 = np.ones((5,5),np.float32)/25
img = cv2.filter2D(src=image, ddepth=-1, kernel = kernel2)
cv2.imshow("orignal", image)
cv2.imshow("Kernel Blue", img)
cv2.waitKey()
cv2.destroyAllWindows()

#blurring an image using built in function
img_blur = cv2.blur(src=image, ksize=(5,5))#using blur function to blur an image where ksize is the kernel size
cv2.imshow("original", image)
cv2.imshow("blurred", img_blur)
cv2.waitKey()
cv2.destroyAllWindows()

#applying gaussian blur to an image
#GaussianBlur(src, ksize, sigmX[, dst[, sigmaY[, borderType]]])
#sigmaX & sigmaY = gaussian kernel standard deviations
gaussian_blur = cv2.GaussianBlur(src=image, ksize=(5,5), sigmaX=0, sigmaY=0)
cv2.imshow("original", image)
cv2.imshow("gaussian blur", gaussian_blur)
cv2.waitKey()
cv2.destroyAllWindows()

#Median Blurring
#medianBlur(src, ksize)
median = cv2.medianBlur(src = image, ksize = 5)
cv2.imshow("original", image)
cv2.imshow("median blur", median)
cv2.waitKey()
cv2.destroyAllWindows()

#Sharpening an image using convolution kernel
# USE TO LEARN MORE ABOUT COMMONLY USED KERNELS: https://en.wikipedia.org/wiki/Kernel_(image_processing)
kernel3 = np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]])
sharp_img = cv2.filter2D(src=image, depth=-1, kernel=kernel3)
cv2.imshow("original", image)
cv2.imshow("sharpened image", sharp_img)
cv2.waitKey()
cv2.destroyAllWindows()

#Bilateral Filtering
#Applies filter to selectively blur similar intensity pixels in a neighborhood
#Lets you control spatial size of filter + degree to which neighboring pixels are included in filtered output
#biltaeralFilter(src, d, sigmaColor, sigmaSpace)
#d = diameter of pixel neighborhood, sigmaColor + sigmapace = color intensity distribution + spatial distribution
bilateral_filter = cv2.bilateralFilter(src=image, d=9, sigmaColor = 75, sigmaSpace = 75)
cv2.imshow("original", image)
cv2.imshow("bilateral filter", bilateral_filter)
cv2.waitKey()
cv2.destroyAllWindows()
