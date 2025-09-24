import cv2
#cropping using openCV
img = cv2.imread('test.jpg')
print(img.shape)
cv2.imshow("original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped_image = img[80:280, 150:330] #slicing to crop the image
# to slice an arry: specify start + end index of the first (# of rows) and second dimension(# of columns)

#display cropped image
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#dividing an image into small patches using cropping
img = cv2.imread("test.jpg")
image_copy = img.copy()
imgheight = img.shape[0]
imgwidth = img.shape[1]
#load height + width to specify range till which the smaller patches need to be cropped out
#patches w/ height:76 px + width: 104 px
M = 76
N = 104
x1 = 0
y1 = 0
for y in range(0, imgheight, M):
    for x in range(0, imgwidth, N):
        if img(imgheight - y) < M or (imgwidth - x) < N:
            break
        y1 = y + M
        x1 = x + N
        #check whether path width or height exceeds img width or height
        if x1 >= imgwidth and y >= imgheight:
            x1 = imgwidth - 1
            y1 = imgheight - 1
            #crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #save each patch into file directory
            cv2.imwrite("tile" + str(x) + "_" + str(y) + ".jpg", tiles)
            cv2.rectangle(img, (x,y), (x1, y1), (0,255,0), 1)
        elif y1>=imgheight: #when patch height exceeds image height
            y1 = imgheight - 1
            #crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #save each patch into file directory
            cv2.imwrite("tile" + str(x) + "_" + str(y) + ".jpg", tiles)
            cv2.rectangle(img, (x,y), (x1, y1), (0,255,0), 1)
        elif x>= imgwidth: #when patch width exceeds image width
            x1 = imgwidth - 1
            #crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #save each patch into file directory
            cv2.imwrite("tile" + str(x) + "_" + str(y) + ".jpg", tiles)
            cv2.rectangle(img, (x,y), (x1, y1), (0,255,0), 1)
        else:
            #crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #save each patch into file directory
            cv2.imwrite("tile" + str(x) + "_" + str(y) + ".jpg", tiles)
            cv2.rectangle(img, (x,y), (x1, y1), (0,255,0), 1)
#save full image into file directory
cv2.imshow("Patched Image", img)
cv2.imwrite("Patched_image.jpg", img)
cv2.waitKey()
cv2.destroyAllWindows
