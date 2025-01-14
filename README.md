My idea
...
Recently, I saw an interesting python library on the Internet for controlling the mouse cursor. I wanted to do some kind of cool project. 
For this project, I decided to use computer vision. The idea is simple - train the YOLO model to search for a palm gesture (to move the mouse) and a fist gesture (to press or hold the left mouse button. 
I decided not to do the right button yet) on the image, and then develop a program that found a gesture in the webcam stream, 
took the coordinates of the center of the building box and correlated them with the coordinates of the mouse, with a fist gesture, the left mouse button would be pressed, but not released. 
