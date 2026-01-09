import cv2
import numpy as np
import pyautogui
import time
from keras.models import load_model
from hand import HandDetector

KEY_MAPPING = {
    'VolumeUp': 'volumeup',
    'VolumeDown': 'volumedown',
    'PlayPause': 'playpause', 
    'Idle': None
}

pyautogui.FAILSAFE = True 
VOLUME_COOLDOWN = 0.1 
ACTION_COOLDOWN = 2.0 

model = load_model('action.h5')
actions = np.array(['VolumeUp', 'VolumeDown', 'PlayPause', 'Idle'])

sequence = []
predictions = []
threshold = 0.8 

last_action_times = {action: 0 for action in actions}

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

print("Controller Started. Move mouse to corner to stop.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)

    img = detector.findHands(img, draw=False)
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
            confidence = res[best_guess_index]
            
            predictions.append(best_guess_index)
            predictions = predictions[-5:]
            most_common_id = np.argmax(np.bincount(predictions))

            final_decision = most_common_id
            if confidence > 0.9:
                final_decision = best_guess_index
            
            if most_common_id == best_guess_index and confidence > threshold:
                current_action = actions[best_guess_index]
                
                last_time = last_action_times[current_action]
                dt = time.time() - last_time 
                
                key_to_press = KEY_MAPPING.get(current_action)

                if key_to_press is not None:
                    
                    required_cooldown = ACTION_COOLDOWN
                    if current_action in ['VolumeUp', 'VolumeDown']:
                        required_cooldown = VOLUME_COOLDOWN
                    
                    if dt > required_cooldown:
                        if '+' in key_to_press:
                            pyautogui.hotkey(*key_to_press.split('+'))
                        else:
                            pyautogui.press(key_to_press)
                        
                        last_action_times[current_action] = time.time()
                        
                        cv2.putText(img, f"SENT: {current_action}", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"ACTION: {current_action}")
                    
                    else:
                        cv2.putText(img, f"WAIT ({required_cooldown - dt:.1f}s)", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                else:
                    cv2.putText(img, f"{current_action}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Feed", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()