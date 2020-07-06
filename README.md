# UAV-Localization-based-on-the-QR-code
This repository stores the code of the localization part of a multi UAVs formation project. 

Dependency: opencv-python

The code is stored in the file: Apriltags_detector_by_me.py, including the detection part and the decoding part.
The detection part can tell the pixel coordinations of four corners of every AprilTag in the image, and the decoding part can tell the id of each tag.
If you want to get the tags' world coordinations, you need to finish the process from the id to coordinations based on the tags map you defined.

My detection process has six steps, they are:
1. Image preprocessing
2. Thresholding and morphological processing (open)
3. Finding the contours
4. Finding the corners in the contours and filter unsatisfied contours
5. Perspective correction and thresholding again
6. Downsampling the tags to 8 x 8

Finally I test the performance of this process compared with the origin algorithm in [here](https://april.eecs.umich.edu/software/apriltag.html).
The origin algorithm is more accurate and faster when the image is very complex or very small (the width is less than 360 pixels), but my code can be far more faster in simple environment (especially in the simulated environment like gazebo). 
