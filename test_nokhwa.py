from typing import List

import camerata

import time
import cv2

cameras = camerata.query(only_usable=True)
print(f"python got {cameras}")


# exit()
cam = camerata.Camera(cameras[0]) # Open a camera
fmts : camerata.CameraFormatOptions = cam.get_format_options()
fmt = fmts[0]

for f in fmts:
    if f.width == 2560 and f.height == 1440:
        fmt = f

cam.open(fmt)
while cam.poll_frame_np() is None: # Note that .poll_frame_* functions never blocks
    time.sleep(0.1) # Wait until we get at least one frame from the camera
#time.sleep(1) # You might want to wait a bit longer while camera is calibrating

img = cam.poll_frame_np()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite("img.png", img)