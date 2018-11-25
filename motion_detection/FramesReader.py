import numpy as np
import cv2
import pyximport; pyximport.install()
import MotionDetectionAlg as md


def read_and_show_frames():
    cap = cv2.VideoCapture('motionDetection.mp4')
    print(cap.isOpened())
    pre_frame = None
    motion = None
    while(True):
        ret, frame = cap.read()
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if pre_frame is not None:
            print("pre frame is None")
            motion = md.detect_motion(pre_frame, current_frame, 30, 6)
        else:
            motion = current_frame
        cv2.imshow('motion', motion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pre_frame = current_frame

    cap.release()
    cv2.destroyAllWindows()


def read_image():
    image = cv2.imread('resmog2.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 2555, cv2.THRESH_BINARY)
    erosionImg = md.erosion_filter(binary,10)
    cv2.imshow('test image',erosionImg)
    cv2.waitKey(0)
    cv2.destoryAllWindows()

def compare_frames():
    frame1 = cv2.imread('frame1.jpg')
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.imread('frame2.jpg')
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    det = md.detect_motion(gray1, gray2, 30, 10)
    cv2.imshow('frames diff', det)
    cv2.waitKey(0)
    cv2.destoryAllWindows()

if __name__ == '__main__':
    read_and_show_frames()


