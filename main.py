"""
To take a picture from the space station:
Download this file and go here:
https://astro-pi-replay-online.astro-pi.org/
press the 'browse' link and opload the file.  
Then press the 'run' button. It should then display a picture taken from the space station.

To try to calculate the speed of the space station:
download the other file named main.py.py and rename it to main.py

the replay can only handle programs named 'main.py'

Then opload the file and run it.  The speed is written in the results.txt file and 
read from the file by the replay tool so that it can be seen on the top left .
The speed is not quite correct but it is a start for you to have a program that proves the conciept. 

Another more informative access is here: 
Go here to the URL shown beneath to navigate to the online test tool under the headline section:
'Accessing the Astro Pi Replay Tool online ' the online tool is called ' Astro Pi Replay Tool' 
 
https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2
Import the Camera class from the picamera-zero module
"""
from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# Capture an image

cam.take_photo("image1.jpg")

