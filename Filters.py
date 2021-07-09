import numpy as np
import cv2
import skvideo
skvideo.setFFmpegPath("C:/ffmpeg/bin/")
import skvideo.io
from Utils import *


"""
 Arithmetic Mean filtering is applied in the video, where in each interaction we find the sum of the
 pixels in a neighborhood area 3x3x3 and we then divide the output with the number of the pixels in 
 that area. 
"""
def ArithmeticMean(A, param):

    A_pad, pad = pad_Checker(A, param)
    output = np.zeros(A_pad.shape)

    for z in range(1, A_pad.shape[0]):
        for i in range(1, A_pad.shape[1]):
            for j in range(1, A_pad.shape[2]):
                output[z,i,j] = np.sum(A_pad[z-1:z+1,i-1:i+1,j-1:j+1])
    output = output * (1 / 27)
    output = output.astype(np.uint8)

    return output

"""
We use Sobel edge detection in our video by specifing  the kernels of horizontal and vertical interaction and
via a 3D convolution operation we applying it in our video. The final result will be the concatenation of the
horizontal and vertical results. 
"""
def mySobel(A, param, strides):

    A_pad, pad = pad_Checker( A, param)

    Gx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # fliping the kernels
    Gx = np.flipud(np.fliplr(Gx))
    Gy = np.flipud(np.fliplr(Gy))

    # extract shape of the video
    xApadShape = A_pad.shape[2]
    yApadShape = A_pad.shape[1]
    zApadShape = A_pad.shape[0]

    # Define the output size
    xOutput = int(((xApadShape - 3 + 2 * pad) / strides) + 1)
    yOutput = int(((yApadShape - 3 + 2 * pad) / strides) + 1)
    zOutput = int(((zApadShape - 3 + 2 * pad) / strides) + 1)
    output = np.zeros((zOutput, yOutput, xOutput))

        #Starting the convolution operator

    for z in range(A_pad.shape[0]):                      #taking notice of Depth column
        if z > A_pad.shape[0]-3:
            break

        for y in range(A_pad.shape[1]):

                if y > A_pad.shape[1] - 3:     # Go to next row once kernel is out of bounds
                    break

                for x in range(A_pad.shape[2]):

                        if x > A_pad.shape[2] - 3:
                            break
                        try:

                            # Making the Convolution operator.
                            gx = np.sum(np.multiply(Gx, A_pad[z, y:y + 3, x:x + 3]))
                            gy = np.sum(np.multiply(Gy, A_pad[z, y:y + 3, x:x + 3]))
                            # concatenate in the same output
                            output[z, y + 1, x + 1] = np.sqrt( gy ** 2 + gx ** 2)
                        except:
                                break
    return output


""" 
Applying Gaussian Blur filtering.
"""
def create_smooth_kernel(size):

    Xsize = size
    Ysize = size
    Zsize = size
    B = np.zeros((Xsize,Ysize,Zsize))
    B.fill(1/(size)**3)
    return B

