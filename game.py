# This code was created by Damian Villarreal-Ayala on April 18 2024
#Imports
import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random

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

class Enemy:
    """
    A class to represent a random circle
    enemy. It spawns randomly within 
    the given bounds.
    """
    def __init__(self):
        self.screen_width = 1250
        self.screen_height = 400
        self.intercepted = False
        self.respawn()
    
    def respawn(self):
        """
        Selects a random location on the screen to respawn
        """
        self.x = self.screen_width - 50
        self.y = random.randint(50, self.screen_height)
    
    def draw(self, image):
        """
        Enemy is drawn as a circle onto the image

        Args:
            image (Image): The image to draw the enemy onto
        """
        cv2.circle(image, (self.x, self.y), 25, RED, 5)

    def check_interception(self, start_point, end_point):
        """
        Check if the enemy is intercepted by the line segment formed by start_point and end_point.

        Args:
            start_point (tuple): The starting point of the line segment
            end_point (tuple): The ending point of the line segment

        Returns:
            bool: True if intercepted, False otherwise
        """
        # Calculate the vector representing the aiming line
        aiming_vector = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])

        # Calculate the vector representing the line segment from enemy's center to the starting point of the aiming line
        enemy_to_start = np.array([start_point[0] - self.x, start_point[1] - self.y])

        # Calculate the dot product of aiming_vector and enemy_to_start
        dot_product = np.dot(aiming_vector, enemy_to_start)

        # Calculate the length of the aiming vector
        aiming_length = np.linalg.norm(aiming_vector)

        # Calculate the perpendicular distance from the enemy's center to the aiming line
        perpendicular_distance = abs(dot_product) / aiming_length

        # If the perpendicular distance is less than the radius of the enemy circle, they intercept
        if perpendicular_distance <= 25:  # Assuming the radius of the enemy circle is 25
            return True
        else:
            return False

class Game:
    def __init__(self):
        # Load game elements
        self.start_time = time.time()

        # Initialize enemies
        self.enemy = Enemy()
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
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

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
            ring = hand_landmarks[HandLandmarkPoints.RING_FINGER_TIP.value]

            # Map the coordinate back to screen dimensions
            pixelCoordindex = DrawingUtil._normalized_to_pixel_coordinates(index.x, index.y, imageWidth, imageHeight)
            pixelCoordmiddle = DrawingUtil._normalized_to_pixel_coordinates(middle.x, middle.y, imageWidth, imageHeight)
            pixelCoordthumb = DrawingUtil._normalized_to_pixel_coordinates(thumb.x, thumb.y, imageWidth, imageHeight)
            pixelCoordring = DrawingUtil._normalized_to_pixel_coordinates(ring.x, ring.y, imageWidth, imageHeight)

            if pixelCoordindex:
                # Draw the circle around the index finger
                cv2.circle(image, (pixelCoordindex[0], pixelCoordindex[1]), 25, GREEN, 5)
            if pixelCoordmiddle:
                # Draw the circle around the middle finger
                cv2.circle(image, (pixelCoordmiddle[0], pixelCoordmiddle[1]), 25, GREEN, 5)        
            if pixelCoordthumb:
                # Draw the circle around the thumb
                cv2.circle(image, (pixelCoordthumb[0], pixelCoordthumb[1]), 25, RED, 5)
            if pixelCoordring:
                # Draw the circle around the ring finger
                cv2.circle(image, (pixelCoordring[0], pixelCoordring[1]), 25, RED, 5)

    # fy_axis_detection was originally written by Damian but edited using ChatGPT when 
    # encountering errors I wasn't familiar with
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
            palm = hand_landmarks[HandLandmarkPoints.MIDDLE_FINGER_MCP.value]
            ring = hand_landmarks[HandLandmarkPoints.RING_FINGER_TIP.value]
            thumb = hand_landmarks[HandLandmarkPoints.THUMB_TIP.value]

            pixelCoordring = DrawingUtil._normalized_to_pixel_coordinates(ring.x, ring.y, imageWidth, imageHeight)
            pixelCoordmiddle = DrawingUtil._normalized_to_pixel_coordinates(middle.x, middle.y, imageWidth, imageHeight)
            pixelCoord_palmpoint = DrawingUtil._normalized_to_pixel_coordinates(palm.x, palm.y, imageWidth, imageHeight)
            pixelCoordthumb = DrawingUtil._normalized_to_pixel_coordinates(thumb.x, thumb.y, imageWidth, imageHeight)

            # If the index finger has a lower y-axis value, it is higher on the screen than the middle finger
            # If it's higher on the screen, the user is likely aiming
            if index.y < middle.y:
                starting_point = (pixelCoordmiddle[0], pixelCoordmiddle[1])
                # The slope between the middle finger and middle finger mcp is used to aim the Line
                slope = self.slope(pixelCoord_palmpoint[0], pixelCoord_palmpoint[1], pixelCoordmiddle[0], pixelCoordmiddle[1])
                
                # The line will always end at the end of the screen, so we only need to change the y-axis
                end_point_y = int(starting_point[1] + (imageWidth - starting_point[0]) * slope)
                end_point = (imageWidth, end_point_y)
                # Draw line from starting_point to end_point
                cv2.line(image, starting_point, end_point, BLUE, 5)
                # Gun will shoot when thumb tip and ring finger tip overlap
                if self.check_shooting(pixelCoordthumb,pixelCoordring) == True:
                        cv2.line(image, starting_point, end_point, RED, 5)
                        return [starting_point, end_point]

                        
    def check_shooting(self, pixelCoordthumb, pixelCoordring):
        if pixelCoordthumb and pixelCoordring:
                    thumb_center = (pixelCoordthumb[0], pixelCoordthumb[1])
                    ring_center = (pixelCoordring[0], pixelCoordring[1])
                    distance = np.linalg.norm(np.array(thumb_center) - np.array(ring_center))
                    if distance < 25: 
                        #print("Overlapping fingers")
                        return True
                    else:
                        return False


    def slope(self, x1, y1, x2, y2):
        # Ensure that the x-values won't be zero when divided
        if x1!=x2:
            return ((y2 - y1) / (x2 - x1))
        else:
            print("Can't provide a slope when the denominator is 0")
            return 
    
                        
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

            # Draw circle around fingers 
            self.finger_detection(image, results, start_time)

            # Draw the enemy/target
            self.enemy.draw(image)
            
            # Checks to see if the user is aiming with their hand
            aiming_line_intercepts = self.fy_axis_detection(image, results)

            # Check if the aiming line intercepts the enemy
            if aiming_line_intercepts:
                interception = self.enemy.check_interception(aiming_line_intercepts[0], aiming_line_intercepts[1])
                if interception:
                    self.enemy.respawn()

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('First Person Shooter', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            
        self.video.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    g = Game()
    g.run()