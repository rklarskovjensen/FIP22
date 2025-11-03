#Go here to the URL shown beneath and navigate to the online  test tool under the headline section:
# 'Accessing the Astro Pi Replay Tool online ' the online tool is called ' Astro Pi Replay Tool' 

# https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2
# Import the Camera class from the picamera-zero module
from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# Capture an image

cam.take_photo("image1.jpg")
