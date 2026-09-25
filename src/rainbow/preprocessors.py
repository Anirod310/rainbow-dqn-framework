import numpy as np
import cv2
from collections import deque

class AtariPreprocessor():
    def __init__(self, frame_stack_size, target_image_size):
        self.frame_stack_size = frame_stack_size
        self.target_image_size = target_image_size
        self.last_raw_frame = None

        self.deque_frames = deque(maxlen=frame_stack_size)

    def reset(self, observation):
        self.deque_frames.clear()
        self.last_raw_frame = observation

        gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_observation = (np.array(cv2.resize(gray_observation, self.target_image_size, interpolation=cv2.INTER_LINEAR), dtype='float32'))/255

        for _ in range(self.frame_stack_size):
           self.deque_frames.append(resized_observation)
        
        return np.stack(self.deque_frames)

    def process(self, observation):

        maxed_observation = np.maximum(self.last_raw_frame, observation)
        self.last_raw_frame = observation

        gray_observation = cv2.cvtColor(maxed_observation, cv2.COLOR_RGB2GRAY)
        resized_observation = (np.array(cv2.resize(gray_observation, self.target_image_size, interpolation=cv2.INTER_LINEAR), dtype='float32'))/255


        self.deque_frames.append(resized_observation)

        return np.stack(self.deque_frames)

