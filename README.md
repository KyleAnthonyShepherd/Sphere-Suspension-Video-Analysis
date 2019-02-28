# Sphere-Suspension-Video-Analysis
Video Processing for Sphere Tracking<br>
<br>
This code tracks the locations of spheres floating inside of a conical tube<br>
<br>
It loads a mp4 file named VID, and performs image analysis to segment each frame<br>
and record the location of the spheres<br>
<br>
If this script is run by itself, the code will process VID.mp4.<br>
First AverageFrame is called to calculate the average frame in the video<br>
Then the ImageProcess function is run over each frame to determine the sphere locations<br>
The ImageProcess is handwritten to perform well on this mp4. Alterations to this<br>
function is required if a differetn video in different lighting conditions is used<br>
<br>
###<br>
Code Written by:<br>
Kyle Shepherd, at Oak Ridge National Laboratory<br>
kas20@rice.edu<br>
June 22, 2018 (Version 1)<br>
Oct 19, 2018 (Version 2, some optimizations)<br>
###<br>
