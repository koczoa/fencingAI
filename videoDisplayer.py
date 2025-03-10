import sys
import cv2
import numpy as np
import torch


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

def skeletonDrawer(array: np.ndarray, footage: np.ndarray) -> None:

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


def main() -> int:
    if len(sys.argv) < 2:
        print(f"No filename provided!")
        return 1
    fileName = sys.argv[1]
    filePath = "resources/" + fileName + ".mp4"
    frameCount = 0
    history = torch.load(f"processed/{fileName}.pt", map_location=torch.device("cpu"))

    footage = cv2.VideoCapture(filename=filePath)
    fps = footage.get(cv2.CAP_PROP_FPS)
    print(fps)
    while footage.isOpened():
        success, frame = footage.read()
        if success:
            currentFrame = history[frameCount].numpy()
            # print(f"current data: {currentFrame}")
            for detection in currentFrame:
                current_person_keypoints = np.delete(detection, slice(5), axis=0)
                skeletonDrawer(current_person_keypoints, frame)

            frameCount += 1
            cv2.imshow("skeletonised", frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord("q"):
                break
        else:
            break

    footage.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())