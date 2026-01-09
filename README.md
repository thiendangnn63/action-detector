Action recognition using LSTM (Long-Short Term Memory)

Collect action using by running `data_collection.py` and recording action by pressing 'S' between sequences to save the action

Current actions include: 
- Hello (Wave 1 hand)
- Thanks (Touching chin then move forward and downward)
- Idle (Do nothing or random movement)

Use `model_training.py` to train the model on the recorded data. I already trained and saved my model `action.h5` with the previously mentioned actions.

Run `action_detect.py` to load the model and use it in real-time.
