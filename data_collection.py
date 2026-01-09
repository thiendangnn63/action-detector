import cv2
import numpy as np
import os
from hand import HandDetector

DATA_ROOT = "MP_Data"
actions = np.array(['Hello', 'Thanks', 'Idle'])
sequences = 30
frames = 30

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

for action in actions:
    for sequence in range(sequences):
        try:
            os.makedirs(os.path.join(DATA_ROOT, action, str(sequence)))
        except OSError:
            pass

for action in actions:
    for sequence in range(sequences):
        for frame in range(frames):
            success, img = cap.read()

            img = cv2.flip(img, 1)

            img = detector.findHands(img)
            landmarks = detector.findPosition(img)

            if frame == 0:
                while True:
                    success, img = cap.read()
                    img = cv2.flip(img, 1)
                    img = detector.findHands(img)

                    cv2.putText(img, 'PRESS "S" TO START', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, f'Current Action: {action} (Seq #{sequence})', (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow('Feed', img)
                    
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
            else:
                cv2.putText(img, f"Sequence: {sequence}/{sequences}", (20, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow("Feed", img)
                cv2.waitKey(1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if landmarks:
                keypoints = landmarks[0]
            else:
                keypoints = [0] * 42

            npy_path = os.path.join(DATA_ROOT, action, str(sequence), str(frame))
            np.save(npy_path, np.array(keypoints))

cap.release()
cv2.destroyAllWindows()