import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase # to use the camera in the browser
import av # to change the camera frames to images and back
import cv2
import math # for geometic calculations
import mediapipe as mp  # Google's library for detecting hands, face, pose

# open cv reads images as BGR
# mediapipe reads images ars RGB

mp_drawing = mp.solutions.drawing_utils  # used to draw lines, points, and landmarks
mp_holistic = mp.solutions.holistic      # detects full body landmarks: face, hands, pose
mp_hands = mp.solutions.hands  # using Hands model instead of Holistic (faster and more accurate for fingers)


#_______________________________________________
# finger counting functions

# compute the angle of (a -- b -- c (b is in the middle and is joint))
def angle3d(a, b, c):
    # check whether a finger joint is straight or bent
    ab = (a.x - b.x, a.y - b.y, a.z - b.z)
    cb = (c.x - b.x, c.y - b.y, c.z - b.z)

    # dot product: measures angle similarity between two vectors
    # find the dot product to find the angle between the two segments
    dot = ab[0]*cb[0] + ab[1]*cb[1] + ab[2]*cb[2]

    # compute magnitude of each vector: |v| = sqrt(x^2 + y^2+ z^2)
    norm_ab = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
    norm_cb = math.sqrt(cb[0]**2 + cb[1]**2 + cb[2]**2)

    # handle invalid cases: dot product rule (if one of the values are 0, the final answer will be 0)
    if norm_ab == 0 or norm_cb == 0: # if segment has zero lengths
        return 0.0 # return vector length

    # acos: based on the output of the cosine, it finds the exact angle
    # ex: cos_val = 0 -> cos: 90 degrees
    # compute the cosine of the angle between ab and cb

    # computers gets slightly different results each time because they uses imperfect binary approximations of numbers,
    # and makes tiny differences in calculations like dot products and acos(), while on paper the numbers are exact
    cos_val = dot / (norm_ab * norm_cb)
     # if due to floading point rounding, the value becomes (ex: 1.000002), it will clamp it back into 1
    cos_val = max(-1.0, min(1.0, cos_val))
    # (math.acos(cos_val)): compute the angle in radians/ math.degrees: converts that angle to degrees
    return math.degrees(math.acos(cos_val))


def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark # ther are 21 hand larndmarks

    # create 2D vector from point a to b: to show hand axis
    def vec2(a, b):
        return (b.x - a.x, b.y - a.y)

    # cmpute the 2D distance between two landmarks
    # (normalize hand size, compare the thumb distance from palm(in general, avoid values that depend on the camera distance))
    def dist(a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx*dx + dy*dy)

    wrist = lm[0] # wrist land mark (the best reference point for almost all orientation)
    middle_mcp = lm[9] # join of middle finger land mark

    # **it uses the hand's internal direction, not screen coordinates**
    vx, vy = vec2(wrist, middle_mcp) # this shows the direction of the hand
    hand_len = math.sqrt(vx*vx + vy*vy) + 1e-6 # compute the magnitude of vector from wrist to middle finger/ add a tiny number to avoid division by zero
    hand_dir = (vx / hand_len, vy / hand_len) # create unique vector (This allows us to project any finger's position onto the hand's natural axis)

    # how far is the point in the direction the hand is pointing
    def proj_on_hand_axis(point):
        vx, vy = vec2(wrist, point)
        return vx * hand_dir[0] + vy * hand_dir[1]

    # defines hand size as distance from wrist to middle MCP
    hand_size = dist(wrist, middle_mcp) + 1e-6

    fingers = 0 # initialize the finger counter

    for tip_idx in [8, 12, 16, 20]:
        pip_idx = tip_idx - 2  # PIP
        mcp_idx = tip_idx - 3  # MCP

        # find 3 landmarks in each finger
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        mcp = lm[mcp_idx]

        proj_tip = proj_on_hand_axis(tip) # how forward the fingertip is
        proj_pip = proj_on_hand_axis(pip) # how forward the PIP joint is (if the finger is more forward, the finger is open)

        # angle between MCP, PIP, TIP
        ang = angle3d(mcp, pip, tip)

        # is the fingere slightly closer to the camera thatn PIP join?
        # smaller z: closer to the camera/ bigger z: farther from the camera
        z_straight = tip.z < pip.z + 0.03 # +0.03 is added because tiny differences between TIP and PIP are NOT real bending
        # so we ignore them
        # only large differences mean the finger is actually bent


#*****************************************
        is_forward = proj_tip > proj_pip + 0.02 * hand_len #The fingertip must be noticeably ahead of the PIP along the direction your hand is pointing
        is_straight = ang > 150  # even if the finger is pointing forward, it must also be straight, not partially bent (human finger n)

        if is_forward and is_straight and z_straight: # 3 conditions to have finger open
            fingers += 1


    # find the center point of the palm in 2D by getting the average
    # to detect thumb direction, find center as reference point
    palm_indices = [0, 1, 5, 9, 13, 17]
    cx = sum(lm[i].x for i in palm_indices) / len(palm_indices)
    cy = sum(lm[i].y for i in palm_indices) / len(palm_indices)

    class point_object:
        pass

    # 3D center of the palm
    # find the thumb tip (is close or far, extend outward or inward)
    palm_center = point_object()
    palm_center.x = cx
    palm_center.y = cy
    palm_center.z = sum(lm[i].z for i in palm_indices) / len(palm_indices)

    # thumb landmarks
    thumb_cmc = lm[1]
    thumb_mcp = lm[2]
    thumb_ip = lm[3]
    thumb_tip = lm[4]

    index_mcp = lm[5]
    ring_mcp = lm[13]
    sx, sy = vec2(ring_mcp, index_mcp) # compute a vector between index_mcp and ring_mcp
    side_len = math.sqrt(sx*sx + sy*sy) + 1e-6 # how long the sideways line is between index MCP and ring MCP to normalize the vector
    side_dir = (sx / side_len, sy / side_len) # measure how far along this sideway direction the thumb tip is

    def proj_on_side_axis(p):
        vx, vy = vec2(palm_center, p)
        return vx * side_dir[0] + vy * side_dir[1]

    thumb_proj = proj_on_side_axis(thumb_tip)
    ip_proj = proj_on_side_axis(thumb_ip)

    # distance between thumb tip and palm center
    tip_palm = dist(thumb_tip, palm_center)

    # angle between thumb MCP and thumb IP
    thumb_angle_ip = angle3d(thumb_mcp, thumb_ip, thumb_tip)
    thumb_angle_mcp = angle3d(thumb_cmc, thumb_mcp, thumb_tip)

    thumb_z_ok = thumb_tip.z < thumb_ip.z + 0.03 # thumb tip must be slightly closer to camera than IP joint
    cond_proj = abs(thumb_proj - ip_proj) > 0.12 * hand_size # thumb must extend outward significantly
    cond_dist = tip_palm > 0.40 * hand_size # thumb tip must be far from palm center
    cond_angle = (thumb_angle_ip > 150) or (thumb_angle_mcp > 150) # thumb must be mostly straight

    if cond_proj and cond_dist and cond_angle and thumb_z_ok:
        fingers += 1

    return fingers


#_______________________________________________
# streamlit app

# the camera is opened in the browser of the person who uses the app (with streamlit-webrtc),
# so the app also works online and not only on my own computer
class FingerCounter(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6, # how sure the model is that it detected something (do I see a hand?)
            min_tracking_confidence=0.5 # how sure the model is that it tracked something (does the hand stay where I think it is?)
        )

    # this function runs for every frame that comes from the camera
    def recv(self, frame):
        # get the frame as RGB for mediapipe
        frame_rgb = frame.to_ndarray(format="rgb24")

        # make detections
        results = self.hands.process(frame_rgb)

        # --------------------------------------------------------
        # Mediapipe Hands DOES NOT provide:
        #   results.right_hand_landmarks
        #   results.left_hand_landmarks
        #
        # Instead it provides:
        #   results.multi_hand_landmarks  -> list of hands
        #   results.multi_handedness      -> list containing "Left" / "Right"
        #
        # We must loop through them together.
        # --------------------------------------------------------
        if results.multi_hand_landmarks:
            # zip() allows us to match each hand with its handedness label
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # get label name ("Left" or "Right")
                label = handedness.classification[0].label

                # count fingers using your function
                fingers = count_fingers(hand_landmarks)

                # choose text position based on which hand it is
                # mediapipe label = real body side
                # but on the screen it looks mirrored, so we swap for display
                # frame_rgb is RGB so colors are (R, G, B)
                if label == "Right":
                    text = f'Left: {fingers}'   # show as Left on screen
                    pos = (10, 80)
                    color = (255, 0, 0)  # red
                else:  # "Left"
                    text = f'Right: {fingers}'  # show as Right on screen
                    pos = (10, 40)
                    color = (0, 0, 255)  # blue

                # draw landmarks on the image
                mp_drawing.draw_landmarks(
                    frame_rgb,  # image to draw on
                    hand_landmarks,  # the 21 hand landmarks
                    mp_hands.HAND_CONNECTIONS,  # how to connect the points
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),  # color of the points
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),  # color of the lines
                )

                # write the finger count text on the image
                cv2.putText(
                    frame_rgb,
                    text,
                    pos,  # position/ 10 pixels from left, 40 or 80 pixels from top
                    cv2.FONT_HERSHEY_SIMPLEX,  # font style
                    1,  # font size
                    color,  # font color (left: red, right: blue)
                    2,  # font thickness
                )

        # send the frame back to the browser to show it
        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")


# online, the browser and the streamlit server can't connect to each other directly,
# so the video goes through a TURN server from metered.ca (free)
# the metered username and password are saved in the streamlit secrets (not on github)
# on my own computer there are no secrets, so only the free google STUN server is used
def get_ice_servers():
    try:
        username = st.secrets["METERED_USERNAME"]
        password = st.secrets["METERED_PASSWORD"]
    except Exception:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return [
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": [
                "turn:global.relay.metered.ca:80",
                "turn:global.relay.metered.ca:80?transport=tcp",
                "turn:global.relay.metered.ca:443",
                "turns:global.relay.metered.ca:443?transport=tcp",
            ],
            "username": username,
            "credential": password,
        },
    ]


st.title("Live Finger Counter")

st.subheader("Show your fingers and count them!")

run = st.checkbox("Click to turn the camera on/off")

# the checkbox turns the camera on/off (so the START, STOP and SELECT DEVICE buttons are not shown)
# when the user turns OFF the camera, the camera is released automatically
webrtc_streamer(
    key="finger-counter",
    desired_playing_state=run,
    video_processor_factory=FingerCounter,
    media_stream_constraints={"video": True, "audio": False}, # only camera, no microphone
    rtc_configuration={"iceServers": get_ice_servers()}, # helps to connect the browser camera to the app online
)


#_______________________________________________
# notes from holistic model
# pass it into the function of cv2 and recolor it from BGR to RGB(openCV uses BGR, holistic uses RGB)
# Initiate holistic model: mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Holistic results: face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
# Recolor image back to BGR for rendering
# print(mp_holistic.FACE_CONNECTIONS)  # This line is not necessary, but it can be used to access face connections if needed.
