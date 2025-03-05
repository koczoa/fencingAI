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
    cv2.line(footage, array[0].astype(int), array[2].astype(int), color, thickness)
    cv2.line(footage, array[2].astype(int), array[4].astype(int), color, thickness)
    cv2.line(footage, array[0].astype(int), array[6].astype(int), color, thickness)
    cv2.line(footage, array[6].astype(int), array[8].astype(int), color, thickness)
    cv2.line(footage, array[8].astype(int), array[10].astype(int), color, thickness)
    # RIGHT SIDE
    color = (255, 0, 0)
    cv2.line(footage, array[1].astype(int), array[3].astype(int), color, thickness)
    cv2.line(footage, array[3].astype(int), array[5].astype(int), color, thickness)
    cv2.line(footage, array[1].astype(int), array[7].astype(int), color, thickness)
    cv2.line(footage, array[7].astype(int), array[9].astype(int), color, thickness)
    cv2.line(footage, array[9].astype(int), array[11].astype(int), color, thickness)
    # MIDDLE
    color = (0, 255, 0)
    cv2.line(footage, array[0].astype(int), array[1].astype(int), color, thickness)
    cv2.line(footage, array[6].astype(int), array[7].astype(int), color, thickness)

def keypointDetection(result, footage):
    for r in result:
        nof = r.keypoints.shape[0]
        for p in range(nof):
            current_person_keypoints = np.delete(r.keypoints.xy[p].numpy(), slice(5), axis=0)
            skeletonMaker(current_person_keypoints, footage)

def main() -> int:
    if len(sys.argv) < 2:
        print(f"No filename provided!")
        return 1

    model = YOLO("yolo11n-pose.pt")
    file = "resources/" + sys.argv[1]


    footage = cv2.VideoCapture(filename=file)
    while footage.isOpened():
        success, frame = footage.read()
        result = model.track(source=frame, save=False, show=False, name="result")
        keypointDetection(result, frame)
        cv2.imshow("skeletonised", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    footage.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())