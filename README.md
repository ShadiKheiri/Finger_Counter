# Finger Counter (Streamlit)

A real-time finger counter built with Python, OpenCV, MediaPipe and Streamlit. It uses the webcam to detect your hands and count how many fingers are open on each hand.

## Live Demo

Try the app here: [Finger Counter Demo](https://finger-counter-app.streamlit.app/)

## Features

- Detects up to two hands at the same time
- Counts the fingers of each hand separately
- Left hand is shown in red and right hand is shown in blue
- Works even when the hand is tilted or rotated
- Uses the camera in the browser, so it works online and locally
- Turn the camera on and off with one checkbox

## How It Works

- MediaPipe Hands finds 21 landmarks on each hand and detects if it is the left or right hand
- The direction of the hand is calculated from the wrist to the middle finger joint
- A finger is counted as open when the fingertip is in front of the middle joint, the finger is straight (angle more than 150 degrees), and it is not bent toward the back
- The thumb has its own rules: it must move away from the palm sideways, be far from the center of the palm, and be mostly straight
- The number of open fingers is written on the video for each hand

## How to Run

Requires Python 3.9 to 3.12 (MediaPipe 0.10.21 does not support newer versions).

```
pip install -r requirements.txt
streamlit run finger_counter.py
```

Then open the app in your browser and click the checkbox to turn the camera on.
