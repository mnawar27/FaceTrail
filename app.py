import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
from flask import send_file


app = Flask(__name__)

nimgs = 10
imgBackground = cv2.imread("static/images/background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load Haarcascade for face detection
face_detector = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_default.xml')

# Create necessary directories
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Initialize attendance file if not already present
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    """Extract faces from an image."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in face extraction: {e}")
        return []

def identify_face(facearray):
    """Identify a face using the trained model."""
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    """Train a face recognition model using KNN."""
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    """Extract attendance records."""
    df = pd.read_csv(attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    """Add a user to the attendance record."""
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)
    if int(userid) not in df['Roll'].tolist():
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

def getallusers():
    """Get all registered users."""
    userlist = os.listdir('static/faces')
    names, rolls = [], []
    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', mess='No trained model found. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', mess='Unable to access camera.')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Unable to grab frame from camera.")
            break

        faces = extract_faces(frame)
        if faces:
            x, y, w, h = faces[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, identified_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return home()

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    user_folder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', mess='Unable to access camera.')

    i, j = 0, 0
    while i < nimgs:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Unable to grab frame from camera.")
            break

        faces = extract_faces(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if j % 5 == 0:
                cv2.imwrite(f'{user_folder}/{newusername}_{i}.jpg', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        
        #cv2.imshow('Adding new User', frame)
        import platform
        if platform.system() != 'Darwin':  # 'Darwin' is the identifier for macOS
            cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()
    return home()

if __name__ == '__main__':
    app.run(debug=True)