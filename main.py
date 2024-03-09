# Importing the necessary libraries
import pickle
import numpy as np
import cv2
import os
from datetime import datetime
import face_recognition
import mysql.connector
from flask import Flask, render_template, Response

# Establishing connection to the MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="firstdb"
)
mycursor = mydb.cursor()

# Setting up the webcam with appropriate height and width
cap = cv2.VideoCapture(0)

# Importing images of students
folder_user_path = 'Images'
user_path = os.listdir(folder_user_path)
student_img_list = [cv2.imread(os.path.join(folder_user_path, i)) for i in user_path]
student_ids = [os.path.splitext(i)[0] for i in user_path]

file = open('encoding_file.p', 'rb')
known_encode_list_with_ids = pickle.load(file)
file.close()
known_encode_list, student_id = known_encode_list_with_ids

last_entry_times = {}

# Function to mark entry in the database
def markEntry(name, table_name):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    if name in last_entry_times and (now - last_entry_times[name]).total_seconds() < 300:
        return

    sql = f"INSERT INTO `{table_name}` (name, entry_time) VALUES (%s, %s)"
    val = (name, dtString)
    mycursor.execute(sql, val)
    mydb.commit()

    last_entry_times[name] = now

# Flask application setup
app = Flask(__name__)

# Route for the video feed
def gen_frames():
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(known_encode_list, encodeFace)
            faceDis = face_recognition.face_distance(known_encode_list, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = student_ids[matchIndex].upper()
                markEntry(name, "registered_user")
            else:
                name = "GUEST"
                markEntry(name, "guest")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (253, 251, 115), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (0, 255, 0), cv2.FILLED)

            cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
