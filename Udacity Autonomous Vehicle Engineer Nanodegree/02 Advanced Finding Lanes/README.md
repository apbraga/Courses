 Course: Udacity Autonomous Car Engineer
 Project 2: Advanced line detection
 Alex Braga

## Overview
This package refer to Project 2 - Advanced Line Detection for Udacity Autonomous Car Engineer Nanodegree.

The objective is to create a pipeline to detect and dress up a video clip with lane annotation in addition to curvature and off center line estimation.

## Import packages

The following packages were used for this project and the implementation was done in Pyhton 3.

```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```

## Camera Calibration

OpenCV function to get camera calibration matrix were implemented used 20 calibration images of chessboard with a 6x9 grid.

```python
def calibrate ():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #   Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Test undistortion on an image
    img = cv2.imread('test_images/test4.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    return mtx, dist

mtx, dist = calibrate()

```

![alt-text-1](output_images/calibrated_chessboard.jpg "chessboard")

![alt-text-1](/test_images/test7.jpg "calibrated_image")



## Tresholding
With a calibrated image we apply color tresholding in the HLS space and gradients using sobel.

```python
def tresholding(img, s_thresh, luv_tres, lab_tres ,sx_thresh):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    l_channel = luv[:,:,0]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(hls_l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    s1_binary = np.zeros_like(scaled_sobel)
    s1_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s2_binary = np.zeros_like(scaled_sobel)
    s2_binary[(l_channel >= luv_tres[0]) & (l_channel <= luv_tres[1])] = 1

    # Threshold color channel
    s3_binary = np.zeros_like(scaled_sobel)
    s3_binary[(b_channel >= lab_tres[0]) & (b_channel <= lab_tres[1])] = 1   

    # Threshold color channel
    s4_binary = np.zeros_like(s_channel)
    s4_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack((s1_binary, s2_binary, s3_binary+s4_binary ))*255
    ret, binary = cv2.threshold(cv2.cvtColor(color_binary,cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)


    return binary

```
![alt-text-1](/output_images/7binary.jpg "title-1")

## Perspective Transformation (Bird's eye view)
Using openCV cv2.warpPerspective function, the region of interest is extracted.
Values are hard coded based on calibrated images and distribution on test images.

```python
def perspective(img):
    img_size = (img.shape[1],img.shape[0])
    src = np.float32([[715,466],[1006,654],[314,656],[575, 466]])
    dst = np.float32([[990,0],[990,720],[290,720],[290,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

```
![alt-text-1](/output_images/7warped.jpg "title-1")

## Lane identification
This stage is divided in 2 cases.

1st case: First frame or Lost track
  Search box is applied by taken the half bottom image and detecting histogram peaks, from the peaks the search box algorithm start all the way up, appending the points related to the line and recentering at each step.

2nd case: Default
  Based on the polynomial from the last frame, it search for lane pixels around the polynomial.

  if it fails, 1st is run.

A polynomial is fit to the points found by case 1 or 2, in case of failure it uses the polynomial from the last frame.

```python
def find_lane_pixels(binary_warped):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            #Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

```


```python
def search_around_poly(warped,left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    #left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255
    #window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

    return leftx, lefty, rightx, righty, out_img

```

```python
def getPoly(warped, poly_left_old, poly_right_old):
    global a
    poly_left_new=np.array([ 0., 0.,  0.])
    poly_right_new=np.array([ 0., 0.,  0.])
    if a == 0:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        poly_left_new=left_fit
        poly_right_new=right_fit
        a=1
    else:

        try:
            leftx, lefty, rightx, righty, out_img = search_around_poly(warped,poly_left_old, poly_right_old)
            1/len(rightx)
            1/len(leftx)
        except:
            leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)


        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            poly_left_new[0] = poly_left_old[0]*0.8 +left_fit[0]*0.2
            poly_left_new[1] = poly_left_old[1]*0.8 +left_fit[1]*0.2
            poly_left_new[2] = poly_left_old[2]*0.8 +left_fit[2]*0.2
            poly_right_new[0] = poly_right_old[0]*0.8 +right_fit[0]*0.2
            poly_right_new[1] = poly_right_old[1]*0.8 +right_fit[1]*0.2
            poly_right_new[2] = poly_right_old[2]*0.8 +right_fit[2]*0.2
            y_lenght = np.linspace(0, warped.shape[0]-1, warped.shape[0])
            left_fitx = poly_left_new[0]*y_lenght**2 + poly_left_new[1]*y_lenght + poly_left_new[2]
            right_fitx = poly_right_new[0]*y_lenght**2 + poly_right_new[1]*y_lenght + poly_right_new[2]
            imgpoints = (y_lenght.astype(int), left_fitx.astype(int),right_fitx.astype(int))
            mask = np.zeros((warped.shape[0],warped.shape[1],3), np.uint8)
            mask[ y_lenght.astype(int) , left_fitx.astype(int)] = [255,255,0]
            mask[ y_lenght.astype(int) , right_fitx.astype(int)] = [255,255,0]

        except:
            poly_left_new = poly_left_old
            poly_right_new = poly_right_old

    return poly_left_new, poly_right_new ,out_img

```

![alt-text-1](/output_images/7lines.jpg "title-1")

## Perspective Transformation
Using openCV cv2.warpPerspective function, the reserve transformation is applied.

```python
def unwarped(fill_image):
    #unwarp lane area images
    src = np.float32([[715,466],[1006,654],[314,656],[575, 466]])
    dst = np.float32([[990,0],[990,720],[290,720],[290,0]])

    M = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(fill_image, M, (fill_image.shape[1],fill_image.shape[0]), flags=cv2.INTER_LINEAR)

    return unwarped

```
![alt-text-1](/output_images/7mask.jpg "title-1")

## Radius and curvature
Using real life x image reference the radius and off centre values are calculated and overlayed into the mask.

```python
def curvature(unwarped,left_fit, right_fit):
    y_lenght = np.linspace(0, unwarped.shape[0]-1, unwarped.shape[0])
    left_fitx = left_fit[0]*y_lenght**2 + left_fit[1]*y_lenght + left_fit[2]
    right_fitx = right_fit[0]*y_lenght**2 + right_fit[1]*y_lenght + right_fit[2]
    #Calculate curvature
    ym_per_pix = 14.4/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = 720
    left_fit_cr = np.polyfit(y_lenght*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_lenght*ym_per_pix, right_fitx*xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    curve=(left_curverad+right_curverad)/2

    if curve > 20000.:
        curve_text = "straight"
    else:
        curve_text = str(int(right_curverad))+" m"

    out = cv2.putText(unwarped, "Radius Curvature: "+ curve_text , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    #off center calculation
    off_center= ((left_fitx[-1]+right_fitx[-1])/2-unwarped.shape[1]/2)*xm_per_pix
    off_center_txt = "Distance from center:" + format(off_center, '.2f') + " m "
    out = cv2.putText(unwarped, off_center_txt, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    return unwarped
```
![alt-text-1](/output_images/7curve.jpg "title-1")

## Pipeline
The complete process is summarized into one  function to apply to the video pipeline.
```python
def process_image(img):
    #Apply distortion transformation for input image
    #img = cv2.imread('test_images/straight_lines1.jpg')
    #plt.imshow(img)
    global a
    global left_fit
    global right_fit
    global poly_left
    global poly_right

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    binary = tresholding(undist, s_thresh=(150, 255), sx_thresh=(50, 100))
    warped = perspective (binary)
    poly_left, poly_right, box = getPoly(warped, poly_left, poly_right)
    mask = lane_area_draw(warped,poly_left, poly_right)
    unwarp = unwarped(mask)
    curve = curvature(unwarp,poly_left, poly_right)
    out_img = cv2.addWeighted(undist, 1.0, curve, 0.5, 0.)
    return out_img

```
![alt-text-1](/output_images/7out_img.jpg "title-1")
## Test Images Pipeline
Generates single image evidence for the purpose of function check.

```python
def generate_examples(fname,idx):
    global a
    global left_fit
    global right_fit
    global poly_left
    global poly_right

    a=0
    poly_left = np.array([ 0., 0.,  0.])
    poly_right = np.array([ 0., 0.,  0.])
    left_fit = np.array([ 0., 0.,  0.])
    right_fit = np.array([ 0., 0.,  0.])
    img = cv2.imread(fname)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    binary = tresholding(undist, s_thresh=(150, 255), sx_thresh=(50, 100))
    warped = perspective (binary)
    poly_left, poly_right, box_search = getPoly(warped, poly_left, poly_right)

    y_lenght = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = poly_left[0]*y_lenght**2 + poly_left[1]*y_lenght + poly_left[2]
    right_fitx = poly_right[0]*y_lenght**2 + poly_right[1]*y_lenght + poly_right[2]
    lines = np.zeros((warped.shape[0],warped.shape[1],3), np.uint8)
    lines[ y_lenght.astype(int) , left_fitx.astype(int)] = [0,255,255]
    lines[ y_lenght.astype(int) , right_fitx.astype(int)] = [0,255,255]
    lines = cv2.addWeighted(lines, 1.0, box_search, 1, 0.)

    mask = lane_area_draw(warped,poly_left, poly_right)
    unwarp = unwarped(cv2.addWeighted(mask, 1.0, lines, 1, 0.))
    curve = curvature(unwarp,poly_left, poly_right)
    out_img = cv2.addWeighted(undist, 1.0, curve, 0.5, 0.)

    cv2.imwrite('output_images/'+str(idx+1)+'test.jpg',img)
    cv2.imwrite('output_images/'+str(idx+1)+'undist.jpg',undist)
    cv2.imwrite('output_images/'+str(idx+1)+'binary.jpg',binary)
    cv2.imwrite('output_images/'+str(idx+1)+'warped.jpg',warped)
    cv2.imwrite('output_images/'+str(idx+1)+'lines.jpg',lines)
    cv2.imwrite('output_images/'+str(idx+1)+'mask.jpg',mask)
    cv2.imwrite('output_images/'+str(idx+1)+'unwarp.jpg',unwarp)
    cv2.imwrite('output_images/'+str(idx+1)+'curve.jpg',curve)
    cv2.imwrite('output_images/'+str(idx+1)+'out_img.jpg',out_img)

images = sorted(glob.glob('test_images/test*.jpg'))
for idx, fname in enumerate(images):
    generate_examples(fname,idx)

```
![alt-text-1](/pipeline.jpeg "title-1")
## Video Pipeline
Apply complete pipeline to a video clip and outputs a processed video.
```python
def processVideo(file):
    global a
    global left_fit
    global right_fit
    global poly_left
    global poly_right
    a=0
    poly_left = np.array([ 0., 0.,  0.])
    poly_right = np.array([ 0., 0.,  0.])
    left_fit = np.array([ 0., 0.,  0.])
    right_fit = np.array([ 0., 0.,  0.])
    white_output = 'process_'+file+'.mp4'
    clip1 = VideoFileClip(file)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    %time white_clip.write_videofile(white_output, audio=False)

processVideo('project_video.mp4')
```
[![](http://img.youtu.be/8h3EM0Q3zeQ.jpg)](http://www.youtu.be/8h3EM0Q3zeQ "Processed Video")

https://www.youtube.com/watch?v=8h3EM0Q3zeQ

## Discussion
Under shadows the line detection became unstable, additional tuning on tresholding is required to make it more suceptible to lighting variation.

Sometimes when lane is not visible on use side it would be good to adjust the impact of polynomial based on the side that has more pixels available for the estimation.
In the project video, the left side estimation provides much more reliable detection than the right, so under a given treshold of number of line pixels on the right side, we can use the left lane to update the polynomial of the right lines.
