import torch
import torch.nn as nn
import torch.optim as optim

class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(21*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Three classes: Unknown, Point Up, Point Down

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = GestureRecognitionModel()

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def preprocess_landmarks(landmarks):
    flattened = []
    for lm in landmarks:
        flattened.extend([lm.x, lm.y, lm.z])
    return torch.tensor(flattened).float().unsqueeze(0)

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = preprocess_landmarks(hand_landmarks.landmark)
                outputs = model(landmarks)
                _, predicted = torch.max(outputs, 1)
                gesture = predicted.item()

                # Map the predicted gesture to a label
                gesture_label = ["Unknown", "Index Point Up", "Index Point Down"][gesture]
                print(gesture_label)

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
