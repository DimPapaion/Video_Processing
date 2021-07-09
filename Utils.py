import numpy as np
import cv2
import skvideo
import skvideo.io

"""
A 3D convolution operation from scratch. As inputs it receives the original video array,
the Kernel, strides and the value of the parameter param which is defining if we wonna apply
padding or not in our input video. 
"""
def myConv3D(A,B,strides,param):
    A_pad,pad=pad_Checker( A,param)
    B = np.flipud(np.fliplr(B))
    # Specifing the x,y,z len of kernel matrix
    xKernShape = B.shape[2]
    yKernShape = B.shape[1]
    zKernShape = B.shape[0]

    # Specifing the x,y,z len of Padded Matrix
    xApadShape = A_pad.shape[2]
    yApadShape = A_pad.shape[1]
    zApadShape = A_pad.shape[0]

    #Definig the output matrix shape.
    xOutput = int(((xApadShape - xKernShape + 2 * pad) / strides) + 1)
    yOutput = int(((yApadShape - yKernShape + 2 * pad) / strides) + 1)
    zOutput = int(((zApadShape - zKernShape + 2 * pad) / strides) + 1)
    output = np.zeros((zOutput,yOutput,xOutput))

    #Starting the convolution operator
    for z in range(A_pad.shape[0]):                      #taking notice of Depth column
        if z > A_pad.shape[0]-zKernShape:
            break

        for y in range(A_pad.shape[1]):

                if y > A_pad.shape[1] - yKernShape:     # Go to next row once kernel is out of bounds
                    break

                for x in range(A_pad.shape[2]):

                        if x > A_pad.shape[2] - xKernShape:
                            break
                        try:
                            # Making the Convolution operator.

                            output[z,y,x] = (B * A_pad[ z:z + zKernShape, y: y + yKernShape,x: x + xKernShape]).sum()

                        except:
                               break

    return output



"""
With this function i am applying zero padding if the param ="same" and when we insert "valid" in param
 we do not apply zero padding in the video. Shapes of kernels should be NxNxN. 
"""
def pad_Checker( A, param=None):
    n=3
    padv = int((n - 1) / 2)
    if param == 'same':
        C = pad_video(A, size=padv)
    elif param == "valid":
        padv = 0
        C = A
    else:
        print("No such value available.! Please select same or valid.!")
    return C, padv


def unpad(A, A_out, size):
    A_unpad = np.zeros((A.shape[0], A.shape[1], A.shape[2]))
    A_unpad[:, :, :] = A_out[size:-size, size:-size, size:-size]
    return A_unpad

def pad_video(A, size):
    A_pad = np.zeros((A.shape[0] + 2*size, A.shape[1] + 2*size, A.shape[2] + 2*size))
    A_pad[1*size:-1*size, 1*size:-1*size, 1*size:-1*size] = A

    return A_pad

def show_vid(name, video):

    cap = cv2.VideoCapture(video)

    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            # we show each frame of the video
            cv2.imshow(name,frame)
            if cv2.waitKey(25) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def Video_writer(A, name):
    writer = skvideo.io.FFmpegWriter(name)
    for i in range(A.shape[0]):
        writer.writeFrame(A[i, :, :])
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    writer.close()
    return
