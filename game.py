# This code was created by Damian Villarreal-Ayala on April 18 2024
#Imports
import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
# from playsound import playsound

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

    def finger_detection(self, image, detection_result, time):
        # Get image details
        imageHeight, imageWidth = image.shape[:2]
            # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
    
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
    
            # Get the coordinate of just the index finger
            index = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            middle = hand_landmarks[HandLandmarkPoints.MIDDLE_FINGER_TIP.value] 
            thumb = hand_landmarks[HandLandmarkPoints.THUMB_TIP.value]

            # Map the coordinate back to screen dimensions
            pixelCoordindex = DrawingUtil._normalized_to_pixel_coordinates(index.x, index.y, imageWidth, imageHeight)
            pixelCoordmiddle = DrawingUtil._normalized_to_pixel_coordinates(middle.x, middle.y, imageWidth, imageHeight)
            pixelCoordthumb = DrawingUtil._normalized_to_pixel_coordinates(thumb.x, thumb.y, imageWidth, imageHeight)

            if pixelCoordindex:
                # Draw the circle around the index finger
                cv2.circle(image, (pixelCoordindex[0], pixelCoordindex[1]), 25, GREEN, 5)
            if pixelCoordmiddle:
                # Draw the circle around the index finger
                cv2.circle(image, (pixelCoordmiddle[0], pixelCoordmiddle[1]), 25, GREEN, 5)        
            if pixelCoordthumb:
                # Draw the circle around the index finger
                cv2.circle(image, (pixelCoordthumb[0], pixelCoordthumb[1]), 25, RED, 5)

    def fy_axis_detection(self, image, detection_result):
        # Get image details
        imageHeight, imageWidth = image.shape[:2]
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Get the coordinates of just the fingers
            index = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            middle = hand_landmarks[HandLandmarkPoints.MIDDLE_FINGER_TIP.value]
            palm_point1 = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_MCP.value]
            palm_point2 = hand_landmarks[HandLandmarkPoints.MIDDLE_FINGER_MCP.value]

            if index.y < middle.y:
                cv2.circle(image, (int(middle.x), int(middle.y)), 25, BLUE, 5)
                #cv2.line(image, (STARTING_X,STARTING_Y), (ENDING_X, ENDING_Y), RED, 25, cv2.LINE_8)
                # Creating two points(tuples) that I'm using to find the slope of the fingers 
                starting_point = (index.x, ((int(middle.y) + int(index.y)) / 2))
                palm_point = (palm_point1.x,((int(palm_point1.y) + int(palm_point2)) / 2))
                cv2.line(image, (int(middle.x), int(middle.y)), (), RED, 25, cv2.LINE_8)


                        
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
            # self.draw_landmarks_on_hand(image, results)

            # Draw box around hand when thumb is extended
            self.finger_detection(image, results, start_time)

            # Checks to see if HandLandmarks 8, 7, 6 are roughly in a line
            # and if 8 has a higher y-value than 12, which indicates that the gun shape is made
            self.fy_axis_detection(image, results)

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