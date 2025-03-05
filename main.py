from ultralytics import YOLO
import sys

def main() -> int:
    model = YOLO("yolo11n-pose.pt")
    result = model.track(source="resources/pelleTest.jpg", save=False, show=True, name="result")
    return 0

if __name__ == '__main__':
    sys.exit(main())