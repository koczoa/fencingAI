from ultralytics import YOLO
import sys
import cv2
import numpy as np


def skeletonMaker(array: np.ndarray, footage: np.ndarray):
    """
    00    Left Shoulder
    01    Right Shoulder
    02    Left Elbow
    03    Right Elbow
    04    Left Wrist
    05    Right Wrist
    06    Left Hip
    07    Right Hip
    08    Left Knee
    09    Right Knee
    10    Left Ankle
    11    Right Ankle
    """

    thickness = 3
    # LEFT SIDE
    color = (0, 0, 255)
    image = cv2.line(img=footage, pt1=array[0].astype(int), pt2=array[2].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[2].astype(int), pt2=array[4].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[0].astype(int), pt2=array[6].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[6].astype(int), pt2=array[8].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[8].astype(int), pt2=array[10].astype(int), color=color,
                     thickness=thickness)
    # RIGHT SIDE
    color = (255, 0, 0)
    image = cv2.line(img=footage, pt1=array[1].astype(int), pt2=array[3].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[3].astype(int), pt2=array[5].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[1].astype(int), pt2=array[7].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[7].astype(int), pt2=array[9].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[9].astype(int), pt2=array[11].astype(int), color=color,
                     thickness=thickness)
    # MIDDLE
    color = (0, 255, 0)
    image = cv2.line(img=footage, pt1=array[0].astype(int), pt2=array[1].astype(int), color=color,
                     thickness=thickness)
    image = cv2.line(img=footage, pt1=array[6].astype(int), pt2=array[7].astype(int), color=color, thickness=thickness)
    return image



def main() -> int:
    model = YOLO("yolo11x-pose.pt")
    testFile = "resources/pelleTest.jpg"
    result = model.track(source=testFile, save=False, show=True, name="result")
    footage = cv2.imread(filename=testFile)

    for r in result:
        nof = r.keypoints.shape[0]
        for p in range(nof):
            current_person_keypoints = np.delete(r.keypoints.xy[p].numpy(), slice(5), axis=0)
            skeletonImage = skeletonMaker(current_person_keypoints, footage)

    cv2.imshow("skeletonised", footage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())