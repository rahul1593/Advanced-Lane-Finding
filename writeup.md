
## Advanced Lane Finding Project
---


### Overview

In this project, I'm going to find the lanes on the road along with the radius of curvature of the road and vehicle position with respect to center of the road.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/cal_op.JPG "Camera Calibration"
[image2]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/cal_test.JPG "Distortion Correction"
[image3]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/thresholded.JPG "Color Mask"
[image4]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/persp_trans.JPG "Output"
[image5]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/persp_trans_thresh.JPG "Output"
[image6]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/lines.JPG "Output"
[image7]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/output.JPG "Output Image"
[image8]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/meta/form.JPG "Formula"
[video1]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/output_videos/project_video.mp4 "Project Video"
[video2]: https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/output_videos/challenge_video.mp4 "Challenge Video"

---

### Camera Calibration

The code for this step is contained in the second code cell of the IPython notebook located in "Project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Here I have assumed the number of corners along x-axis as 9 and along y-axis as 6. Few chessboard images for calibration which have less than these number of corners along respective axis will not be used for calibration.

### Pipeline (single images)

#### 1. Distortion correction in images

After calculating the camera matrix and distortion coefficients, I used them for undistorting the test images. Following is the test image which was undistorted using these calcuated parameters:
![alt text][image2]

#### 2. Identifying the lane lines on road images

Color thresholds for each of red, green and blue color are decide by using the following formula:

$$
colorThreshold = regularColorThreshold - \frac{(regularMedianColor - imageColorMedian)}{effectFactor}
$$

Here regularColorThreshold is the threshold color value chosen for an image having median color equal to regularMedianColor approximately. imageColorMedian is the median of a color in current image. effectFactor is used to decide the effect of difference between color median of current and standard image on the regular threshold value.

Following image shows the actual image and color thresholded image:

![alt text][image3]

This formula could help to calibrate the color intensities moderately in different lighting conditions. This works well for the project video.

But for challenge video, color thresholding doesn't work that well, so I used gradients along x-axis, HSL color format and RGB color difference to get the lane lines.

#### 3. Perspective transform

The code for my perspective transform is present in cell no 7. The function `getSourceDestPoints` takes the input image as argument and returns `source` and `destination` points for that image. This function call is followed by the use of `cv2.getPerspectiveTransform(src, dst)` function to transform the perspective and `cv2.warpPerspective`  function to get the warped image.

![alt text][image4]

Following is the source and destination points calculation code(cell no. 2):

```python
def getSourceDestPoints(image_2d_mask):
    # two slots in image
    w = image_2d_mask.shape[1]
    h = image_2d_mask.shape[0]
    
    src = np.float32(([(w*2/5)+33, h*0.65],
    [(w*3/5), h*0.65],
    [w, h],
    [0, h]))
    
    dst = np.float32([[0, 0],
        [w, 0],
        [w, h],
        [0, h]])

    return src, dst
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 545, 468      | 0, 0          | 
| 768, 468      | 1280, 0       |
| 1280, 720     | 1280, 720     |
| 0, 720        | 0, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Identifying lane-line pixels and fit their positions with a polynomial

Then I calculated the sum of all values in y-axis for each value in x-axis to get the positions of starting points of left and right lanes from the bottom of the image. I used these values as starting point for performing sliding window search for the lane lines in the warped binary image. I got all the values for left and right lane which were part of the line and then used `numpy.polyfit` method to fit a second order polynomial to these values. Code for this is present in cell no 8.

Following image shows the line drawn after polynomial fit:

![alt text][image6]

#### 5. Calculating radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell no 10. For calculating radius of curvature, I used polynomial coefficients derived by using the fitting `x` and `y` coordinated for left and right lane after scaling the pixel values to meters. I used the following formula to calculate radius of curvature:

![alt text][image8]


Where, `y` is the maximum value corresponding to the bottom of the image. `A` and `B` are coefficients derived after polyfit on scaled values in meters.

Code for finding vehicle position is in cell no 12. I used the difference between average of base position of x-fits for left and right lanes and comparing them with the half of image width.

#### 6. Output Image with plotted lane area

I implemented this step in cell no 9. Here is an example of my result on a test image after getting radius of curvature and vehicle position:

![alt text][image7]

---

### Pipeline (video)

#### Video Output.

For running the pipeline for project video, I made a few basic changes. Following are the changes that I made:
1. Make some variables as global, so that they may be used in case of bad frames.
2. Use weighted averaging for getting starting position of the lane lines for smoother lines.

Here's a [link to my video result](https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/output_videos/project_video.mp4)

Here's a [link to challenge video result](https://github.com/rahul1593/Advanced-Lane-Finding/raw/master/output_videos/challenge_video.mp4)

---

### Discussion

#### 1. Current Shortcomings of pipeline

The major issue in the current pipeline is the detection of lanes in varying environment. The image detection technique used in project and challenge video is different and do not work very well if interchanged.

If there are other vehicles on the same lane, this pipeline is very likely to fail.

#### 2. Possible Improvements in pipeline

Lane lines detection in varying scenarios can be further improved. Also, predictions of turn and possibility of lane could be made in case of bad frames instead of just re-using the data from the previous frames. This would help in case of multiple subsequent bad frames.

