import cv2
import numpy as np
from keras.models import load_model
from hand import HandDetector

actions = np.array(['Hello', 'Thanks', 'Idle'])

model = load_model('action.h5')

sequence = []
sentence = []
threshold = 0.8

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
        print(sequence)

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            best_guess_index = np.argmax(res)
            confidence = res[best_guess_index]

            if confidence > threshold:
                current_action = actions[best_guess_index]
                cv2.putText(img, f'Current action: {current_action} | Confidence: {confidence}%',
                            (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    else:
        cv2.putText(img, 'No hand', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('Detect', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()