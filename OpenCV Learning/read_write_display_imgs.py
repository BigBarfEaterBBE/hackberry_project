#import cv2 library
import cv2
#cv2.imread() is used to read an image
img_greyscale = cv2.imread('test.jpg', 0) #flag 0 = greyscale, flag 1 = color w/ alpha, flag -1 = color w/ alpha
#cv2.imshow() displays an image in a window
cv2.imshow('grayscale image', img_greyscale) #cv2.imshow('window name', image variable)
#cv2.waitKey() waits for a key press to close window and 0 specifies indefinite loop
cv2.waitKey(0)
#cv2.destroyAllWindows() closes all windows opened
cv2.destroyAllWindows()
#cv2.imwrite() is used to write an image
cv2.imwrite('grayscale.jpg', img_greyscale)