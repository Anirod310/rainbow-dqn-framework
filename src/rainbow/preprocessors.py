import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

class AtariPreprocessor():
    def __init__(self, frame_stack_size, target_image_size):
        self.frame_stack_size = frame_stack_size
        self.target_image_size = target_image_size

        self.deque_frames = deque(maxlen=frame_stack_size)

    def reset(self, observation):
        self.deque_frames.clear()

        gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_observation = cv2.resize(gray_observation, self.target_image_size, interpolation=cv2.INTER_LINEAR)

        for _ in range(self.frame_stack_size):
           self.deque_frames.append(resized_observation)

        return np.stack(self.deque_frames)
        

preprocessor = AtariPreprocessor(4, (84, 84))

image = cv2.imread("breakout.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

processed_image = preprocessor.reset(image)

print(processed_image.shape)
print(np.array_equal(processed_image[0], processed_image[1]))
print(np.array_equal(processed_image[1], processed_image[2]))
print(np.array_equal(processed_image[2], processed_image[3]))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_image[0], cmap="gray")
plt.title("Processed")
plt.axis("off")

plt.show()