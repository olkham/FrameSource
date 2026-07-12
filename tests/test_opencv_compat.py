# Test OpenCV compatibility with FrameSourceFactory
from framesource import FrameSourceFactory
import cv2

def main(framework: str = "opencv"):
    """Run OpenCV compatibility tests."""

    print(f"\nTesting OpenCV-compatible API with framework: {framework}")

    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

    if framework.lower() == "opencv":
        cap = cv2.VideoCapture(0)
    elif framework.lower() == "framesource":
        cap = FrameSourceFactory.create('webcam', source_id=0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera")
            break

        cv2.imshow("Test", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print("Exiting...")
            break

        if key == ord('='):
            exp = cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"Current exposure: {exp}")
            cap.set(cv2.CAP_PROP_EXPOSURE, exp + 1)
        elif key == ord('-'):
            exp = cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"Current exposure: {exp}")
            cap.set(cv2.CAP_PROP_EXPOSURE, exp - 1)

    # Release the camera when done
    cap.release()


if __name__ == "__main__":
    main(framework="opencv")
    main(framework="framesource")
