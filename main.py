from ultralytics import YOLO
import sys
import cv2
import numpy as np


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


def positionRecognition(array: np.ndarray):
    pass

def skeletonDrawer(array: np.ndarray, footage: np.ndarray):

    def drawSegment(x: int, y:int, color: (int, int, int)) -> None:
        try:
            if not array[x].any() or not array[y].any():
                return
            thickness = 3
            cv2.line(footage, array[x].astype(int), array[y].astype(int), color, thickness)
        except IndexError:
            pass

    # LEFT SIDE
    red = (0, 0, 255)
    drawSegment(0, 2, red)
    drawSegment(2, 4, red)
    drawSegment(0, 6, red)
    drawSegment(6, 8, red)
    drawSegment(8, 10, red)

    # RIGHT SIDE
    blue = (255, 0, 0)
    drawSegment(1, 3, blue)
    drawSegment(3, 5, blue)
    drawSegment(1, 7, blue)
    drawSegment(7, 9, blue)
    drawSegment(9, 11, blue)

    # MIDDLE
    green = (0, 255, 0)
    drawSegment(0, 1, green)
    drawSegment(6, 7, green)

def keypointDetection(result, footage):
    for r in result:
        if r.keypoints.xy[0].numpy().size == 0:
            return
        nof = r.keypoints.xy.shape[0]
        print(f"number of people detected: {nof}")
        for p in range(nof):
            current_person_keypoints = np.delete(r.keypoints.xy[p].numpy(), slice(5), axis=0)
            skeletonDrawer(current_person_keypoints, footage)

def main() -> int:
    if len(sys.argv) < 3:
        print(f"No filename or yolo size provided!")
        return 1
    size = sys.argv[1]
    model = YOLO(f"yolo11{size}-pose.pt")
    file = "resources/" + sys.argv[2]


    footage = cv2.VideoCapture(filename=file)
    while footage.isOpened():
        success, frame = footage.read()
        if success:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            result = model.track(source=frame, save=False, show=False, name="result", persist=True, tracker="botsort.yaml")
            keypointDetection(result, frame)
            cv2.imshow("skeletonised", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
         # if cv2.waitKey(10) & 0xFF == ord('q'):
         #    break


    footage.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())