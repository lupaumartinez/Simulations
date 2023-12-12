
import numpy as np

def spiral_cw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0])        # take first row
        A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)

def spiral_ccw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0][::-1])    # first row reversed
        A = A[1:][::-1].T         # cut off first row and rotate clockwise
    return np.concatenate(out)

def base_spiral(nrow, ncol, clock_type):

    if clock_type == 'cw':

        spiral = spiral_cw(np.arange(nrow*ncol).reshape(nrow,ncol))[::-1]

    elif  clock_type == 'ccw':

        spiral = spiral_ccw(np.arange(nrow*ncol).reshape(nrow,ncol))[::-1]
   
    return spiral

def to_spiral(A, clock_type):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape, clock_type)] = A.flat
    return B

def from_spiral(A):
    A = np.array(A)
    return A.flat[base_spiral(*A.shape, clock_type)].reshape(A.shape)

def matrix_time(exposure_time, number_pixel):

    x = number_pixel
    y = number_pixel
    matrix_pixel_time = (np.arange(x*y).reshape(x,y) + 1)*exposure_time

    return matrix_pixel_time.T