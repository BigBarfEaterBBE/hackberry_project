import cv2
import numpy as np

#read image
image = cv2.imread('image.jpg')
#image.shape returns height, width, # of channels
h,w,c = image.shape
print("Original Heigh & Width:", h,"x",w)
#resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
# src: input image, disize: desired size, fx: scale along x-axis, fy, scale along y-axis, interpolation: options of different methods of resize
#resize by specifying width + height
down_width = 300
down_height = 200
down_points = (down_width, down_height)
#resize image
resize_down = cv2.resize(image, down_points, interpolation = cv2.INTER_LINEAR)

up_width = 600
up_height = 400
up_points = (up_width, up_height)
#resize image
resized_up = cv2.resize(image, up_points, interpolation = cv2.INTER_LINEAR)
#display
cv2.imshow('Resized Down by defining height + width', resize_down)
cv2.waitKey()
cv2.imshow('Resized Up by defining height + width', resized_up)
cv2.waitKey()
cv2.destroyAllWindows()

#Resizing with scaling factor
#scaling up image 1.2 times by specifying both scaling factors
scale_up_x = 1.2
scale_up_y = 1.2
#scaling down image 0.6 times specifying a single scale factor
scale_down = 0.6

scaled_f_down = cv2.resize(image, None, fx = scale_down, fy = scale_down, interpolation = cv2.INTER_LINEAR)
scaled_f_up = cv2.resize(image, None, fx = scale_up_x, fy = scale_up_y, interpolation = cv2.INTER_LINEAR)
#display
cv2.imshow('Resized Down by defining scaling factor', scaled_f_down)
cv2.waitKey()
cv2.imshow('Resized Up by defining scaling factor', scaled_f_up)
cv2.waitKey()
cv2.destroyAllWindows()

#Resizing with different interpolation methods:
# INTER_AREA = pixel area relation for resampling, best for reducing size
# INTER_CUBIC = bicubic interpolation for resizing, acts on 4x4 neighboring pixels of image and takes weights average of the 16 pixels to create the new interpolated pixel
# INTER_LINEAR = similar to INTER_CUBIC but uses 2x2 neighboring pixels to get weighted average
# INTER_NEAREST = uses only one neighboring pixel from image for interpolation
res_inter_nearest = cv2.resize(image, None, fx = scale_down, fy = scale_down, interpolation = cv2.INTER_NEAREST)
res_inter_linear = cv2.resize(image, None, fx = scale_down, fy = scale_down, interpolation = cv2.INTER_LINEAR)
res_inter_area = cv2.resize(image, None, fx = scale_down, fy= scale_down, interpolation = cv2.INTER_AREA)
#concatenate images in horizontal axis for comparison
vertical = np.concatenate((res_inter_nearest, res_inter_linear, res_inter_area), axis = 0)
#display
cv2.imshow('Inter Nearest :: Inter Linear :: Inter Area', vertical)
cv2.waitKey()
cv2.destroyAllWindows()