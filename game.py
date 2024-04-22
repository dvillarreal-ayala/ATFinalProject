# This code was created by Damian Villarreal-Ayala on April 18 2024
#Imports
import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time

# Library Constants (from FingerTracking)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Game:
    def __init__(self):
        # Load game elements
        self.start_time = time.time()

        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        self.video = cv2.VideoCapture(0)

    def draw_landmarks_on_hand(self, image, detection_result):
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                    hand_landmarks_proto,
                                    solutions.hands.HAND_CONNECTIONS,
                                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                                    solutions.drawing_styles.get_default_hand_connections_style())
            
    def run(self):
        # Begin writing code
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes in mirrored - flip it
            image = cv2.flip(image, 1)

            # Keep track of time
            start_time = time.time()

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Hand Tracking', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
            
        self.video.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    g = Game()
    g.run()