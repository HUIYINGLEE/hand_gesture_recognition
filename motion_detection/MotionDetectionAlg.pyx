import numpy as np

# step1: mark the difference changed between two frame/images and do thresholding
def frames_diff(gray_frame1, gray_frame2, diffThreshold):
    height, width = gray_frame1.shape[:2]
    print(gray_frame1.shape)
    diff = np.zeros((height, width)) # 0 is black, 1 is white
    front = 255

    for x in range(width):
        for y in range(height):
            obs_diff = abs(int(gray_frame1[y,x]) - int(gray_frame2[y,x]))
            if obs_diff >= diffThreshold:
                diff[y,x] = front

    return diff

# step2: erosion filter
def erosion_filter(frame, noiseFilterSize):
    n = (noiseFilterSize-1)//2# filter window size has to be odd
    # initialize a erosion filter kernal
    kernal = np.ones((n, n))
    height, width = frame.shape[:2]
    # add padding to the frame
    padding = n//2
    paddedFrame = np.zeros((height+2*padding,width+2*padding))
    #trun the frame to [0,1] in case it is [0, 255]
    frame = (frame == 255).astype(int)

    paddedFrame[padding:-padding, padding:-padding] = frame
    #initialize the output frameS
    outputFrame = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            outputFrame[y,x] = 255 if np.add(paddedFrame[y:y+n, x:x+n], kernal).sum() == 2*n**2 else 0
    return outputFrame

def detect_motion(gray_frame1, gray_frame2, diffThreshold, noiseFilterSize):
    diff = frames_diff(gray_frame1, gray_frame2, diffThreshold)
    detected = erosion_filter(diff, noiseFilterSize)
    return detected


