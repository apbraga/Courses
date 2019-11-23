# **Finding Lane Lines on the Road**

## Writeup

Alex Braga | alexbraga101@gmail.com

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.

* Step 1 : Select region of interest
* Step 2 : Change to gray colorspace
* Step 3 : Segment based on color intensity
* Step 4 : Smooth image with filter
* Step 5 : Get edges mask using Canny
* Step 6 : Get lines using Hough
* Step 7 : Draw line on start image
Repeat for each frame

In order to draw a single line on the Step 7 the following steps were taken:

Loop each line from Hough
* Step 1 : calculate slope, bias and X for y = height*2/3
* Step 2 : Decide if line consisted of left or right traffic lane based on the slope
* Step 3 : Accumulate slope and X
End Loop
* Step 4 : Calculate average slope and X
* Step 5 : Draw average lines

### 2. Identify potential shortcomings with your current pipeline

The region of interest may be an issue depending on the data, basically it works for the data used in this activity, but is not generic enough to adapt for other video sources.

Segmentation based on color intensity of grayscale image have huge impact on the final result, it does no adapt well enough to shadows or day time.


### 3. Suggest possible improvements to your pipeline

One potential improvement would be to perform the Segmentation based on color instead of intensity, HSV colorspace could be used instead of BRG bringing better accuracy under conditions with shadow.
