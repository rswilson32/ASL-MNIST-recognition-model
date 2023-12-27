from flask import Flask, render_template, Response
import cv2
import threading
from camera_recognition import Net, classify

app = Flask(__name__)
net = Net()
net.load_weights('cnn156.pt')

camera = cv2.VideoCapture(0)
thread = None
lock = threading.Lock()

def classify_and_display(frame):
    frame = cv2.flip(frame, 1)
    # Define the region of interest (ROI)
    top, right, bottom, left = 50, 350, 300, 600
    roi = frame[top:bottom, right:left]
    roi = cv2.flip(roi, 1)

    # Convert to gray, add Gaussian Blur (makes blurry)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Classify with CNN
    letter = classify(gray, net)

    # Displaying box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_PLAIN

    # Adding prediction to the top of the box
    text_position = (left - 10, top - 10)
    cv2.putText(frame, letter, text_position, font, 3, (0, 0, 255), 2)

    return frame

def generate_video_feed():
    global camera
    while True:
        with lock:
            ret, frame = camera.read()
        if not ret:
            break

        frame = classify_and_display(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_frames():
    global camera
    while True:
        with lock:
            ret, _ = camera.read()
        if not ret:
            break

if __name__ == '__main__':
    thread = threading.Thread(target=capture_frames)
    thread.start()
    app.run(debug=True, use_reloader=False)

