import cv2
import mediapipe as mp
import serial

class HandGestureDetector:
    def __init__(self, serial_port):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        # Initialize Serial Communication
        self.serial_port = serial.Serial(serial_port, 9600)

    def detect_gesture(self, landmarks):
        # Index finger tip and PIP joint
        index_finger_tip = landmarks[8]
        index_finger_pip = landmarks[6]

        if index_finger_tip.y < index_finger_pip.y:
            return 1  # Index Point Up
        elif index_finger_tip.y > index_finger_pip.y:
            return 2  # Index Point Down
        else:
            return 0  # Unknown Gesture

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        gesture = 0  # Default to unknown

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks.landmark)

        return frame, gesture

    def annotate_frame(self, frame, gesture):
        if gesture == 1:
            message = "Index Point Up"
        elif gesture == 2:
            message = "Index Point Down"
        else:
            message = "Unknown Gesture"
        
        cv2.putText(frame, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def send_serial_cmd(self, gesture):
        if gesture == 1:
            command = "ss001x"
        elif gesture == 2:
            command = "ss002x"
        else:
            return  # No command for unknown gesture

        self.serial_port.write(command.encode())

class HandGestureApp:
    def __init__(self, serial_port):
        self.detector = HandGestureDetector(serial_port)
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, gesture = self.detector.process_frame(frame)
            frame = self.detector.annotate_frame(frame, gesture)
            self.detector.send_serial_cmd(gesture)

            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    serial_port = "COM3"  # Replace with your serial port
    app = HandGestureApp(serial_port)
    app.run()
