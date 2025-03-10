import numpy as np
from ultralytics import YOLO
import sys
import cv2


def keypointDetection(result, history) -> None:
    for r in result:
        keypointsRelevant = r.keypoints.xy.numpy()
        # print(keypointsRelevant)
        if keypointsRelevant[0].size == 0:
            filler = np.zeros((1, 17, 2))
            history.append(np.array(filler))
        else:
            history.append(keypointsRelevant)

def main() -> int:
    if len(sys.argv) < 3:
        print(f"No filename or yolo size provided!")
        return 1

    size = sys.argv[1]
    model = YOLO(f"yolo11{size}-pose.pt")
    fileName = sys.argv[2]
    filePath = "resources/" + fileName + ".mp4"
    frameCount = 0
    history = []

    footage = cv2.VideoCapture(filename=filePath)
    while footage.isOpened():
        success, frame = footage.read()
        if success:
            result = model.track(source=frame, save=False, show=False, name="result", persist=True, tracker="botsort.yaml")
            keypointDetection(result, history)
            frameCount += 1
        else:
            break
    with open(f"processed/{fileName}_saveDump", "w") as f:
        for line in history:
            f.write(f"{line}\n")

    np.save(f"processed/{fileName}.npy", np.array(history))
    print(frameCount)
    footage.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())