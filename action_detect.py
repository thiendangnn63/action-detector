import cv2
import numpy as np
from keras.models import load_model
from hand import HandDetector

actions = np.array(['VolumeUp', 'VolumeDown', 'PlayPause', 'Screenshot', 'CloseCurrentWindow', 'Idle'])

model = load_model('action.h5')

sequence = []
sentence = []
predictions = []
threshold = 0.6

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    landmarks = detector.findPosition(img)

    if len(landmarks) > 0:
        sequence.append(landmarks[0])
        sequence = sequence[-30:]

        if len(sequence) == 30:
            seq = np.array(sequence)

            velocity = np.diff(seq, axis=0)
            zeros = np.zeros((1, 42))
            final = np.vstack([zeros, velocity])
            
            res = model.predict(np.expand_dims(final, axis=0), verbose=0)[0]
            best_guess_index = np.argmax(res)

            predictions.append(best_guess_index)
            predictions = predictions[-10:]

            most_common_id = np.argmax(np.bincount(predictions))

            if most_common_id == best_guess_index:
                if res[best_guess_index] > threshold:
                    current_action = actions[best_guess_index]
                    cv2.putText(img, f'Current action: {current_action}',
                                (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, f'Confidence: {(res[best_guess_index] * 100):.2f}%',
                                (10, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    else:
        cv2.putText(img, 'No hand', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow('Detect', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()