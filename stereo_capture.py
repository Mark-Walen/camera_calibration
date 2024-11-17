import cv2


def capture_stereo(queue_func, height=480, width=640):
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        while True:
            ret, frame = cap.read()
            print(frame.shape)
            if not ret:
                break
            cv2.imshow('right frame', frame)
            lframe = frame[:, :width]
            rframe = frame[:, width:]
            if queue_func:
                queue_func(lframe, rframe)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
    except Exception as e:
        print("capture_stereo: exception {}".format(e))
    finally:
        cap.release()


if __name__ == '__main__':
    capture_stereo(None)
