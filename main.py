
import cv2
import mediapipe as mp
import math
import winsound
import threading
import time

# -------------------------------
# CAMERA SETUP
# -------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("DRIVER SAFETY MONITORING SYSTEM STARTED")

# -------------------------------
# FACE MESH SETUP
# -------------------------------

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# -------------------------------
# VARIABLES
# -------------------------------

closed_frames = 0

alarm_active = False
alarm_lock = threading.Lock()

driving_start_time = time.time()

BREAK_INTERVAL = 3600  # 1 hour
break_reminder_shown = False

sos_alert_shown = False

# -------------------------------
# EYE RATIO FUNCTION
# -------------------------------

def eye_ratio(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append((x, y))

    vertical1 = math.dist(points[1], points[5])
    vertical2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 1.0

    return (vertical1 + vertical2) / (2 * horizontal)


# -------------------------------
# ALARM FUNCTION
# -------------------------------

def play_alarm():

    global alarm_active

    with alarm_lock:

        if alarm_active:
            return

        alarm_active = True

    try:
        end_time = time.time() + 300  # 5 minutes

        while time.time() < end_time:
            winsound.Beep(2000, 700)
            time.sleep(0.2)

    finally:
        with alarm_lock:
            alarm_active = False


# -------------------------------
# MAIN PROGRAM
# -------------------------------

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Camera reading failed")
            break

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        status = "FACE NOT DETECTED"

        # -------------------------------
        # FACE DETECTION
        # -------------------------------

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]
            landmarks = face.landmark

            left_ratio = eye_ratio(
                landmarks, LEFT_EYE, width, height
            )

            right_ratio = eye_ratio(
                landmarks, RIGHT_EYE, width, height
            )

            ratio = (left_ratio + right_ratio) / 2

            # -------------------------------
            # EYE DETECTION
            # -------------------------------

            if ratio < 0.20:
                closed_frames += 1
            else:
                closed_frames = 0
                sos_alert_shown = False

            if closed_frames >= 15:

                status = "DROWSINESS ALERT!"

                if not alarm_active:
                    threading.Thread(
                        target=play_alarm,
                        daemon=True
                    ).start()

                # SOS DEMO ALERT
                if not sos_alert_shown:
                    print("EMERGENCY SOS ALERT!")
                    sos_alert_shown = True

            else:
                status = "EYES OPEN"

            # Draw eye points
            for index in LEFT_EYE + RIGHT_EYE:

                x = int(landmarks[index].x * width)
                y = int(landmarks[index].y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (255, 0, 0),
                    -1
                )

        else:
            closed_frames = 0
            status = "FACE NOT DETECTED"

        # -------------------------------
        # DRIVING BREAK REMINDER
        # -------------------------------

        driving_time = time.time() - driving_start_time

        minutes = int(driving_time // 60)
        seconds = int(driving_time % 60)

        if driving_time >= BREAK_INTERVAL:

            if not break_reminder_shown:

                print("PLEASE TAKE A BREAK!")
                winsound.Beep(1200, 1000)

                break_reminder_shown = True

            cv2.putText(
                frame,
                "PLEASE TAKE A BREAK!",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2
            )

        # -------------------------------
        # DISPLAY INFORMATION
        # -------------------------------

        cv2.putText(
            frame,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255) if closed_frames >= 15
            else (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Monitoring Time: {minutes}:{seconds:02d}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Driver Safety Monitoring System",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# -------------------------------
# CLOSE CAMERA
# -------------------------------

camera.release()
cv2.destroyAllWindows()

print("Driver Safety Monitoring System Stopped")
