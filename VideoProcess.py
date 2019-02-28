'''
Video Processing for Sphere Tracking

This code tracks the locations of spheres floating inside of a conical tube

It loads a mp4 file named VID, and performs image analysis to segment each frame
and record the location of the spheres

If this script is run by itself, the code will process VID.mp4.
First AverageFrame is called to calculate the average frame in the video
Then the ImageProcess function is run over each frame to determine the sphere locations
The ImageProcess is handwritten to perform well on this mp4. Alterations to this
function is required if a differetn video in different lighting conditions is used

###
Code Written by:
Kyle Shepherd, at Oak Ridge National Laboratory
kas20@rice.edu
June 22, 2018 (Version 1)
Oct 19, 2018 (Version 2, some optimizations)
###
'''

#### Import BLock ####
# the import block imports needed modules, and spits out a json file with
# version numbers so the code can be repeatable
file = open("ModuleVersions.json", 'w')
modules = {}

import sys
modules['Python'] = dict([('version', sys.version_info)])

import json
modules['json'] = dict([('version', json.__version__)])

import PIL
from PIL import Image
modules['PIL'] = dict([('version', PIL.__version__)])

import numpy
modules['numpy'] = dict([('version', numpy.__version__)])

import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
modules['matplotlib'] = dict([('version', matplotlib.__version__)])

import skimage
from skimage.filters import threshold_minimum
from skimage import measure
modules['skimage'] = dict([('version', skimage.__version__)])

import scipy
modules['scipy'] = dict([('version', scipy.__version__)])

import os

import imageio
modules['imageio'] = dict([('version', imageio.__version__)])

json.dump(modules, file, indent=4, sort_keys=True)
file.close()
#### END Import Block ####

def ImageProcess(C,SaveHisto=True):
    '''
This function performs the image processing on an given image C

Inputs:
###
C: The image to perform the image analysis
Format: 2D numpy array, unint8

SaveHisto: Switch to output the image histogram before binary segmentation
If True, the code takes longer to run due to time needed to output the image histogram
###

The function outputs a processed frame in the folder MOVIE
White area is where the sphere was detected
    '''
    C[numpy.where(C<0)]=0 # all darker regions get flattened to zero

    C=numpy.mean(C,axis=2) # get average value of RGB, greyscale conversion
    PicData=C.flatten()
    histo,histoedge=numpy.histogram(PicData,bins=255,range=(0,255))

    # histogram has a downward slope, like an exponential decay.
    # a good threshold for segmentation is when the downward trend reverses
    # for the 3rd time, +10 to the found value
    T = histo[11:]-histo[10:-1]
    loc=numpy.where(T>0)[0]
    thresh=loc[numpy.min([2,loc.shape[0]-1])]+10

    # thresh=numpy.argmin(histo[125:numpy.argmax(histo)])+125
    if SaveHisto:
        HistoPrint(histoedge,histo,thresh)

    # Implement the threshold
    C[numpy.where(C<thresh)]=0
    C[numpy.where(C>=thresh)]=1
    img=C.astype(numpy.bool_)

    # using mathematical morphology to smooth the image
    img=scipy.ndimage.morphology.binary_dilation(img).astype(img.dtype)
    img=scipy.ndimage.morphology.binary_dilation(img).astype(img.dtype)
    img=scipy.ndimage.morphology.binary_dilation(img).astype(img.dtype)
    img=scipy.ndimage.morphology.binary_erosion(img).astype(img.dtype)
    img=scipy.ndimage.morphology.binary_erosion(img).astype(img.dtype)
    img=scipy.ndimage.morphology.binary_erosion(img).astype(img.dtype)

    # label bright regions, remove spots smaller than 30 pixels in area
    PicLabel = img.astype(numpy.bool_)
    all_labels = measure.label(PicLabel)
    ObjectSize=scipy.ndimage.measurements.sum(PicLabel,all_labels,index=numpy.arange(numpy.max(all_labels)+1))
    Noise=numpy.where(ObjectSize<30)[0]

    for NR in Noise:
        img[numpy.where((all_labels==NR))]=0

    # saves a picture of the processed frame
    picture=Image.fromarray(img.astype('uint8')*255)
    picture.save('MOVIE/test'+str(i)+'.png')

    # prints coordinates of the sphere locations
    # finds the center of the remaining bright spots
    # not perfect, if the two spheres overlapped, they are only counted as 1
    # and if noise filtration was not perfect, 3 or more spheres could be counted.

    # to implment later, a K-means clustering algorithm could find 2 and exactly 2 spheres in each frame
    Spheres=numpy.where(ObjectSize>=30)[0]
    for S in Spheres:
        y,x=numpy.where((all_labels==S))
        f.write(str(numpy.mean(y))+'\n')
        f.flush()
        os.fsync(f)

def HistoPrint(histoedge,histo,thresh):
    '''
saves histogram to an image file. Was used to determine the threshold equation

Inputs:
###
histoedge: The edges of the histogram bins calculated from numpy.histogram()
Format: 1D numpy array

histo: the values of the histogram bins calculated from numpy.histogram()
Format: 1D numpy array

thresh: The location of the binary segmentation threshhold
A cross is plotted on the histogram at the location of the threshhold
Format: integer
###

The function outputs an histogram as an image
    '''
    plt.rc('xtick',labelsize=6)
    plt.rc('ytick',labelsize=6)
    plt.rc('axes',titlesize=8)
    dx=5
    dy=3
    fig = plt.figure(figsize=(dx, dy))
    plot = fig.add_axes([.75/dx,.5/dy,4/dx,2/dy])

    plot.plot(histoedge[0:-1],histo)
    plot.plot(thresh,histo[thresh],'+')
    plot.set_title('Histogram, with marked threshold')
    plot.set_ylabel('Count')
    plot.set_xlabel('Intensity')
    plot.set_xlim(0,255)
    # plot.set_xticks(numpy.arange(130,160,2))
    plot.set_ylim(0,500)
    fig.savefig('Histograms/'+str(i),dpi=1000)  #savefig, don't show
    plot.clear()
    plt.close()

def AverageFrame(reader,StartFrame,Verbose=True):
    '''
Finds the average frame of an MP4 file
It adds all the frames together, then divides each value by the number of frames

Inputs:
###
reader: the reader object from imageio.get_reader()
Example: reader=imageio.get_reader('VID.mp4',fps=30)

StartFrame: Which frame to start the averaging
Some MP4 files have a weird start where the first 5 frames or so are identical
Format: integer
###

The function outputs the average frame as an image named BASE
    '''
    BASE=reader.get_data(StartFrame).astype('uint32')
    TotalFrames=reader.get_length()
    for i in range(StartFrame+1,reader.get_length()):
        if Verbose:
            print('frames remaining = ' + str(TotalFrames-i))
        BASE=BASE+reader.get_data(i)
    BASE=BASE.astype('float64')
    BASE=BASE/(reader.get_length()-StartFrame)
    picture=Image.fromarray(BASE.astype('uint8'))
    # saves average frame
    picture.save('BASE.png')

if __name__ == "__main__":
    #switch to print histograms. Used to get inital threshold value equation
    StartFrame=5
    #create necessary folders
    if not os.path.exists('Histograms'):
        os.makedirs('Histograms')
    if not os.path.exists('MOVIE'):
        os.makedirs('MOVIE')

    # load video to be python accessible
    reader = imageio.get_reader('VID.mp4',fps=30)

    # Finds average frame
    print(reader.get_length())
    AverageFrame(reader,StartFrame,Verbose=True)
    BASE=Image.open('BASE.png')
    BASE=numpy.array(BASE).astype('int8')
        # we want negative values for processing purposes, not just unsigned

    # open file for measuring sphere height
    f=open('heights.csv','w')

    # begin image analysis loop
    for i in range(StartFrame+1,reader.get_length()): # start at 6 due to weird initial stationary frames

        print('Analyzing frame '+str(i))
        B=reader.get_data(i) # get the ith frame
        C=B.astype('int8')-BASE # get the difference between the current frame and the average frame
        ImageProcess(C,SaveHisto=False)

    # combines processed images into a human viewible movie
    writer = imageio.get_writer('SphereTracking.mp4', fps=30)
    filenames=os.listdir("./MOVIE")
    for i in range(6,reader.get_length()):
        print('writing frame'+str(i))
        filename='test'+str(i)+'.png'
        writer.append_data(imageio.imread('MOVIE/'+filename))
