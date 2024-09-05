from skimage.filters import threshold_local # to add a kind of black and white feel to the scanned image
import numpy as np
import cv2
import imutils

# load the image
image=  cv2.imread("images/questions.jpg")
orig = image.copy() # keep a copy as well

# resize the image - focus on the contours
height= image.shape[0] # image.shape returns a tuple -  (height, width, channels). Extract height
width =image.shape[1] # Extract width/ number of columns in the image
ratio =0.2 # rescaled to 20% of its original dimensions 

# new width and height
width = int(width*ratio)
height =int(height*ratio)

# resized image with new dimensions
image = cv2.resize(image,(width, height))

# grayscale the image to reduce color
# converts the image from the BGR color space (Blue, Green, Red) to grayscale 
grey_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image by applying a gaussian blur to the image 
grey_scaled = cv2.GaussianBlur(grey_scaled, (5,5),0) # (5,5) is the size of the kernel, which determines the amount of blurring

# Now that the image is smooth and has less noise, we can detect edges
edged= cv2.Canny(grey_scaled, 50,200) # canny edge detection algorithm

# # Display the original and edge-detected images
# cv2.imshow("Image", image) # DISPLAYS THE RESIZED IMAGE in a window titles "Image"
# cv2.waitKey(0) # Waits indefinitely for a key press before closing the window.
# cv2.imshow("Edges detected", edged)
# cv2.waitKey(0)


# find those contours
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # extracting top 5 largest contours 

# Draw the contours on the image
# contour_image = image.copy() # create a copy of the image to draw contours on
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# # Display the image with contours
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)

# looping over the list of contours to find the one that represents the boundary 
for contour in contours:
    perimeter = cv2.arcLength(contour, True) # calculate the perimeter 
    # approximate your contour
    approximation = cv2.approxPolyDP(contour, 0.02*perimeter, True)
    
    # if our contour has 4 points, then surely, it should be the paper(rectangle)
    if len(approximation) == 4:
        paper_outline = approximation # then its the boundary 
        break

# Draw the found contour.
# cv2.drawContours(image,[paper_outline],-1,(0,255,0),2)
# cv2.imshow("Found outline", image)
# cv2.waitKey(0)

# we want a 90-degree view of the image especially if it is tilted
# Identify the 4 edge points of the image and arrange it as to where we think it should be

# 4 edge points - top left, top right, bottom right, bottom left
# sum of the coordinates (x,y) is largest for top right and smallest for bottom left

#  largest difference of points is the Top left Corner
#  smallest difference of points is the Bottom left Corner

def arrange_points_in_space(points):
    # initialize a list of co-ordinates that will be ordered
    # first entry is top-left point, second entry is top-right
    # third entry is bottom-right, forth/last point is the bottom left point.
    # a 4 by 2 matrix because we have 4 edge points and each has an x and y coordinate
    rectangle = np.zeros((4,2), dtype = "float32")

    sum_points= points.sum(axis =1) # Calculate the sum of the x and y coordinates for each point.
    # bottom left/rectangle[3] point should be the smallest sum

    rectangle[0] = points[np.argmin(sum_points)]
    rectangle[2] = points[np.argmax(sum_points)]

    #bottom right will have the smallest difference
    #top left will have the largest difference.
    diff_points = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff_points)] 
    rectangle[3] = points[np.argmax(diff_points)]

    return rectangle

def set_four_points(image, points):
    # get the order of the points
    # The points are unpacked into variables top_left, top_right, bottom_right, and bottom_left
    rectangle = arrange_points_in_space(points)
    (top_left, top_right, bottom_right, bottom_left) = rectangle

    # compute dimensions of the new arranged rectangle using the euclidean distance
    left_height = np.sqrt(((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2))
    top_width = np.sqrt(((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2))
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2))

    # find the distances 
    max_height= max(int(left_height), int(right_height))
    max_width = max(int(top_width), int(bottom_width))

    destination=np.array([
        [0,0],
        [max_width-1, 0],
        [max_width -1, max_height-1], 
        [0, max_height -1 ]

    ], dtype= "float32")

    # calculates the transformation matrix that maps the rectangle points to the destination points.
    matrix = cv2.getPerspectiveTransform(rectangle, destination)

    # apply the perspective transformation matrix to the image to get the warped image
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped

# rescaled the paper outline back to its original by removing the ratio scaling
warped = set_four_points(orig, paper_outline.reshape(4, 2) * (1/ratio))

# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) # for grey scaled image
# threshold = threshold_local(warped, 11, offset=10, method="gaussian") # convert grayscale image into a binary image
# warped = (warped > threshold).astype("uint8") * 255 # binary conversion by comparing the grayscale image to the local threshold and then converting the result to an 8-bit image.

# show the original and scanned images
print("Image Reset in progress")
cv2.imshow("Original", cv2.resize(orig, (width, height)))
cv2.imshow("Scanned", cv2.resize(warped, (width, height)))
cv2.waitKey(0)