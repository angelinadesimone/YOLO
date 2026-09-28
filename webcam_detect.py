import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check the camera connection or permissions.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            result = model(frame, verbose=False)[0]
            cv2.imshow("YOLOv8 Small - Webcam Detection", result.plot())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
