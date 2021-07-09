import numpy as np
import skvideo
import skvideo.io
skvideo.setFFmpegPath("C:/ffmpeg/bin/")
from Utils import *
from Filters import *

def main():

    # ============================================#
    #        Reading the Video from disc          #
    # ============================================#

    vid = skvideo.io.vread("Resources/video.mp4")
    A_or = np.empty_like(vid[..., 0])

    # ============================================#
    #        Transform it in GrayScale            #
    # ============================================#

    for i in range(vid.shape[0]):
        A_or[i] = cv2.cvtColor(vid[i], cv2.COLOR_RGB2GRAY)

    # ============================================#
    #        Apply Arithmetic Mean Filtering      #
    # ============================================#

    Arith_Mean = ArithmeticMean(A_or, param="same")
    Video_writer(Arith_Mean,"Results/Arithmetic_Mean.mp4")
    show_vid(name="Arithmetic_Mean", video="Results/Arithmetic_Mean.mp4")

    # ============================================#
    #        Apply Sobel Edge Detection           #
    # ============================================#

    Sobel = mySobel(A_or, param="same", strides=1)
    Video_writer(Sobel, "Results/Sobel_Edge_Detection.mp4")
    show_vid(name="Sobel_Edge_Detection", video="Results/Sobel_Edge_Detection.mp4")

    # ============================================#
    #        Apply Sobel Edge Detection           #
    # ============================================#

    K = create_smooth_kernel(size=3)
    Gaus = myConv3D(A_or, K, strides=1, param="same")
    Video_writer(Gaus, "Results/Gaussian_Blur.mp4")
    show_vid(name="Gaussian_Blur", video="Results/Gaussian_Blur.mp4")

if __name__ == "__main__":
    main()