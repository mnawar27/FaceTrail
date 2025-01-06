# import cv2
# import os
# from flask import Flask, request, render_template
# from datetime import date
# from datetime import datetime
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import joblib

# app = Flask(__name__)

# nimgs = 10

# imgBackground=cv2.imread("background.png")

# datetoday = date.today().strftime("%m_%d_%y")
# datetoday2 = date.today().strftime("%d-%B-%Y")


# # face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_detector = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_default.xml')



# if not os.path.isdir('Attendance'):
#     os.makedirs('Attendance')
# if not os.path.isdir('static'):
#     os.makedirs('static')
# if not os.path.isdir('static/faces'):
#     os.makedirs('static/faces')
# if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
#     with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
#         f.write('Name,Roll,Time')

# def totalreg():
#     return len(os.listdir('static/faces'))

# def extract_faces(img):
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
#         return face_points
#     except:
#         return []

# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)


# def train_model():
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             resized_face = cv2.resize(img, (50, 50))
#             faces.append(resized_face.ravel())
#             labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces, labels)
#     joblib.dump(knn, 'static/face_recognition_model.pkl')

# def extract_attendance():
#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     names = df['Name']
#     rolls = df['Roll']
#     times = df['Time']
#     l = len(df)
#     return names, rolls, times, l

# def add_attendance(name):
#     username = name.split('_')[0]
#     userid = name.split('_')[1]
#     current_time = datetime.now().strftime("%H:%M:%S")

#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     if int(userid) not in list(df['Roll']):
#         with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
#             f.write(f'\n{username},{userid},{current_time}')

# def getallusers():
#     userlist = os.listdir('static/faces')
#     names = []
#     rolls = []
#     l = len(userlist)

#     for i in userlist:
#         name, roll = i.split('_')
#         names.append(name)
#         rolls.append(roll)

#     return userlist, names, rolls, l


# @app.route('/')
# def home():
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# @app.route('/start', methods=['GET'])
# def start():
#     names, rolls, times, l = extract_attendance()

#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

#     ret = True
#     cap = cv2.VideoCapture(0)
#     while ret:
#         ret, frame = cap.read()
#         if len(extract_faces(frame)) > 0:
#             (x, y, w, h) = extract_faces(frame)[0]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
#             identified_person = identify_face(face.reshape(1, -1))[0]
#             add_attendance(identified_person)
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#             cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#             cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
#         imgBackground[162:162 + 480, 55:55 + 640] = frame
#         cv2.imshow('Attendance', imgBackground)
#         if cv2.waitKey(1) == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



# @app.route('/add', methods=['GET', 'POST'])
# def add():
#     newusername = request.form['newusername']
#     newuserid = request.form['newuserid']
#     userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
#     if not os.path.isdir(userimagefolder):
#         os.makedirs(userimagefolder)
#     i, j = 0, 0
#     cap = cv2.VideoCapture(0)
#     while 1:
#         _, frame = cap.read()
#         faces = extract_faces(frame)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
#             cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
#             if j % 5 == 0:
#                 name = newusername+'_'+str(i)+'.jpg'
#                 cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
#                 i += 1
#             j += 1
#         if j == nimgs*5:
#             break
#         cv2.imshow('Adding new User', frame)
#         if cv2.waitKey(1) == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     print('Training Model')
#     train_model()
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# if __name__ == '__main__':
#     app.run(debug=True)

# # import cv2
# # import os
# # from flask import Flask, request, render_template, Response
# # from datetime import date
# # from datetime import datetime
# # import numpy as np
# # from sklearn.neighbors import KNeighborsClassifier
# # import pandas as pd
# # import joblib

# # app = Flask(__name__)

# # nimgs = 10

# # imgBackground = cv2.imread("background.png")

# # datetoday = date.today().strftime("%m_%d_%y")
# # datetoday2 = date.today().strftime("%d-%B-%Y")

# # face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # if not os.path.isdir('Attendance'):
# #     os.makedirs('Attendance')
# # if not os.path.isdir('static'):
# #     os.makedirs('static')
# # if not os.path.isdir('static/faces'):
# #     os.makedirs('static/faces')
# # if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
# #     with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
# #         f.write('Name,Roll,Time')

# # def totalreg():
# #     return len(os.listdir('static/faces'))

# # def extract_faces(img):
# #     try:
# #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #         face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
# #         return face_points
# #     except:
# #         return []

# # def identify_face(facearray):
# #     model = joblib.load('static/face_recognition_model.pkl')
# #     return model.predict(facearray)

# # def train_model():
# #     faces = []
# #     labels = []
# #     userlist = os.listdir('static/faces')
# #     for user in userlist:
# #         for imgname in os.listdir(f'static/faces/{user}'):
# #             img = cv2.imread(f'static/faces/{user}/{imgname}')
# #             resized_face = cv2.resize(img, (50, 50))
# #             faces.append(resized_face.ravel())
# #             labels.append(user)
# #     faces = np.array(faces)
# #     knn = KNeighborsClassifier(n_neighbors=5)
# #     knn.fit(faces, labels)
# #     joblib.dump(knn, 'static/face_recognition_model.pkl')

# # def extract_attendance():
# #     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
# #     names = df['Name']
# #     rolls = df['Roll']
# #     times = df['Time']
# #     l = len(df)
# #     return names, rolls, times, l

# # def add_attendance(name):
# #     username = name.split('_')[0]
# #     userid = name.split('_')[1]
# #     current_time = datetime.now().strftime("%H:%M:%S")

# #     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
# #     if int(userid) not in list(df['Roll']):
# #         with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
# #             f.write(f'\n{username},{userid},{current_time}')

# # def getallusers():
# #     userlist = os.listdir('static/faces')
# #     names = []
# #     rolls = []
# #     l = len(userlist)

# #     for i in userlist:
# #         name, roll = i.split('_')
# #         names.append(name)
# #         rolls.append(roll)

# #     return userlist, names, rolls, l


# # @app.route('/')
# # def home():
# #     names, rolls, times, l = extract_attendance()
# #     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# # @app.route('/start', methods=['GET'])
# # def start():
# #     names, rolls, times, l = extract_attendance()

# #     if 'face_recognition_model.pkl' not in os.listdir('static'):
# #         return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

# #     ret = True
# #     cap = cv2.VideoCapture(0)
# #     while ret:
# #         ret, frame = cap.read()
# #         if len(extract_faces(frame)) > 0:
# #             (x, y, w, h) = extract_faces(frame)[0]
# #             cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
# #             cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
# #             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
# #             identified_person = identify_face(face.reshape(1, -1))[0]
# #             add_attendance(identified_person)
# #             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
# #             cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
# #             cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
# #             cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
# #             cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        
# #         imgBackground[162:162 + 480, 55:55 + 640] = frame
# #         # Instead of using cv2.imshow(), we will stream the image to the browser.
# #         _, jpeg = cv2.imencode('.jpg', imgBackground)
# #         img_bytes = jpeg.tobytes()
# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n')

# #     cap.release()


# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(start(), mimetype='multipart/x-mixed-replace; boundary=frame')


# # @app.route('/add', methods=['GET', 'POST'])
# # def add():
# #     newusername = request.form['newusername']
# #     newuserid = request.form['newuserid']
# #     userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
# #     if not os.path.isdir(userimagefolder):
# #         os.makedirs(userimagefolder)
# #     i, j = 0, 0
# #     cap = cv2.VideoCapture(0)
# #     while 1:
# #         _, frame = cap.read()
# #         faces = extract_faces(frame)
# #         for (x, y, w, h) in faces:
# #             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
# #             cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
# #             if j % 5 == 0:
# #                 name = newusername+'_'+str(i)+'.jpg'
# #                 cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
# #                 i += 1
# #             j += 1
# #         if j == nimgs*5:
# #             break
# #         # Use video feed for the adding user section as well
# #         _, jpeg = cv2.imencode('.jpg', frame)
# #         img_bytes = jpeg.tobytes()
# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n')
# #     cap.release()
# #     train_model()
# #     names, rolls, times, l = extract_attendance()
# #     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# # if __name__ == '__main__':
# #     app.run(debug=True)

###before matplot
# import cv2
# import os
# from flask import Flask, request, render_template
# from datetime import date, datetime
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import io
# from flask import send_file


# app = Flask(__name__)

# nimgs = 10
# imgBackground = cv2.imread("static/images/background.png")

# datetoday = date.today().strftime("%m_%d_%y")
# datetoday2 = date.today().strftime("%d-%B-%Y")

# # Load Haarcascade for face detection
# face_detector = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_default.xml')

# # Create necessary directories
# os.makedirs('Attendance', exist_ok=True)
# os.makedirs('static', exist_ok=True)
# os.makedirs('static/faces', exist_ok=True)

# # Initialize attendance file if not already present
# attendance_file = f'Attendance/Attendance-{datetoday}.csv'
# if not os.path.exists(attendance_file):
#     with open(attendance_file, 'w') as f:
#         f.write('Name,Roll,Time\n')

# def totalreg():
#     return len(os.listdir('static/faces'))

# def extract_faces(img):
#     """Extract faces from an image."""
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
#         return face_points
#     except Exception as e:
#         print(f"Error in face extraction: {e}")
#         return []

# def identify_face(facearray):
#     """Identify a face using the trained model."""
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)

# def train_model():
#     """Train a face recognition model using KNN."""
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             resized_face = cv2.resize(img, (50, 50))
#             faces.append(resized_face.ravel())
#             labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces, labels)
#     joblib.dump(knn, 'static/face_recognition_model.pkl')

# def extract_attendance():
#     """Extract attendance records."""
#     df = pd.read_csv(attendance_file)
#     names = df['Name']
#     rolls = df['Roll']
#     times = df['Time']
#     l = len(df)
#     return names, rolls, times, l

# def add_attendance(name):
#     """Add a user to the attendance record."""
#     username, userid = name.split('_')
#     current_time = datetime.now().strftime("%H:%M:%S")

#     df = pd.read_csv(attendance_file)
#     if int(userid) not in df['Roll'].tolist():
#         with open(attendance_file, 'a') as f:
#             f.write(f'{username},{userid},{current_time}\n')

# def getallusers():
#     """Get all registered users."""
#     userlist = os.listdir('static/faces')
#     names, rolls = [], []
#     for user in userlist:
#         name, roll = user.split('_')
#         names.append(name)
#         rolls.append(roll)
#     return userlist, names, rolls, len(userlist)

# @app.route('/')
# def home():
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# @app.route('/start', methods=['GET'])
# def start():
#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         return render_template('home.html', mess='No trained model found. Please add a new face to continue.')

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         return render_template('home.html', mess='Unable to access camera.')

#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Error: Unable to grab frame from camera.")
#             break

#         faces = extract_faces(frame)
#         if faces:
#             x, y, w, h = faces[0]
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
#             identified_person = identify_face(face.reshape(1, -1))[0]
#             add_attendance(identified_person)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#             cv2.putText(frame, identified_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
#         imgBackground[162:162 + 480, 55:55 + 640] = frame
#         cv2.imshow('Attendance', imgBackground)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return home()

# @app.route('/add', methods=['GET', 'POST'])
# def add():
#     newusername = request.form['newusername']
#     newuserid = request.form['newuserid']
#     user_folder = f'static/faces/{newusername}_{newuserid}'
#     os.makedirs(user_folder, exist_ok=True)

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         return render_template('home.html', mess='Unable to access camera.')

#     i, j = 0, 0
#     while i < nimgs:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Error: Unable to grab frame from camera.")
#             break

#         faces = extract_faces(frame)
#         for x, y, w, h in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
#             cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
#             if j % 5 == 0:
#                 cv2.imwrite(f'{user_folder}/{newusername}_{i}.jpg', frame[y:y+h, x:x+w])
#                 i += 1
#             j += 1
        
#         #cv2.imshow('Adding new User', frame)
#         import platform
#         if platform.system() != 'Darwin':  # 'Darwin' is the identifier for macOS
#             cv2.imshow('Adding new User', frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     train_model()
#     return home()

# if __name__ == '__main__':
#     app.run(debug=True)

# import cv2
# import os
# from flask import Flask, request, render_template, Response, redirect, url_for, jsonify
# from datetime import date, datetime
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import joblib

# app = Flask(__name__)

# nimgs = 10
# imgBackground = cv2.imread("static/images/background.png")

# datetoday = date.today().strftime("%m_%d_%y")
# datetoday2 = date.today().strftime("%d-%B-%Y")

# # Load Haarcascade for face detection
# face_detector = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_default.xml')

# # Create necessary directories
# os.makedirs('Attendance', exist_ok=True)
# os.makedirs('static', exist_ok=True)
# os.makedirs('static/faces', exist_ok=True)

# # Initialize attendance file if not already present
# attendance_file = f'Attendance/Attendance-{datetoday}.csv'
# if not os.path.exists(attendance_file):
#     with open(attendance_file, 'w') as f:
#         f.write('Name,Roll,Time\n')

# def totalreg():
#     return len(os.listdir('static/faces'))

# def extract_faces(img):
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
#         return face_points
#     except:
#         return []

# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)

# def train_model():
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             resized_face = cv2.resize(img, (50, 50))
#             faces.append(resized_face.ravel())
#             labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces, labels)
#     joblib.dump(knn, 'static/face_recognition_model.pkl')

# def extract_attendance():
#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     names = df['Name']
#     rolls = df['Roll']
#     times = df['Time']
#     l = len(df)
#     return names, rolls, times, l

# def add_attendance(name):
#     username = name.split('_')[0]
#     userid = name.split('_')[1]
#     current_time = datetime.now().strftime("%H:%M:%S")

#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     if int(userid) not in list(df['Roll']):
#         with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
#             f.write(f'\n{username},{userid},{current_time}')

# def getallusers():
#     userlist = os.listdir('static/faces')
#     names = []
#     rolls = []
#     l = len(userlist)

#     for i in userlist:
#         name, roll = i.split('_')
#         names.append(name)
#         rolls.append(roll)

#     return userlist, names, rolls, l


# @app.route('/')
# def home():
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# @app.route('/start', methods=['GET'])
# def start():
#     names, rolls, times, l = extract_attendance()

#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

#     ret = True
#     cap = cv2.VideoCapture(0)
#     while ret:
#         ret, frame = cap.read()
#         if len(extract_faces(frame)) > 0:
#             (x, y, w, h) = extract_faces(frame)[0]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
#             identified_person = identify_face(face.reshape(1, -1))[0]
#             add_attendance(identified_person)
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#             cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#             cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        
#         imgBackground[162:162 + 480, 55:55 + 640] = frame
#         _, jpeg = cv2.imencode('.jpg', imgBackground)
#         img_bytes = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(start(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/add', methods=['GET', 'POST'])
# def add():
#     if request.method == 'POST':
#         newusername = request.form['newusername']
#         newuserid = request.form['newuserid']
#         userimagefolder = f'static/faces/{newusername}_{newuserid}'

#         if not os.path.isdir(userimagefolder):
#             os.makedirs(userimagefolder)
        
#         # Start face capture process in a separate thread or after the POST request is done
#         return redirect(url_for('capture', newusername=newusername, newuserid=newuserid))

#     return render_template('add_user.html')
    
# @app.route('/capture', methods=['GET', 'POST'])
# def capture():
#     if request.method == 'POST':
#         newusername = request.form.get('newusername')
#         newuserid = request.form.get('newuserid')
#         return redirect(url_for('capture', newusername=newusername, newuserid=newuserid))
    
#     newusername = request.args.get('newusername')
#     newuserid = request.args.get('newuserid')
#     return f"Captured Username: {newusername}, Captured UserID: {newuserid}"

# import os
# import base64
# from flask import Flask, render_template, request, redirect, url_for
# from io import BytesIO
# from PIL import Image


# @app.route('/')
# def index():
#     return render_template('index.html')  # This should be your template

# @app.route('/upload', methods=['POST'])
# def upload():
#     image_data = request.form.get('image')  # Get the base64 string from the form
#     if image_data:
#         # Remove the "data:image/png;base64," part from the string
#         image_data = image_data.split(',')[1]
        
#         # Decode the base64 string
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
        
#         # Save the image to a file
#         image.save('captured_image.png')

#         return 'Image uploaded and saved successfully'
    

# @app.route('/add_user', methods=['GET', 'POST'])
# def add_user():
#     if request.method == 'POST':
#         # This endpoint will handle the image capture after face scan
#         image_data = request.form.get('image')
#         if image_data:
#             # Process the base64 image data
#             image_data = image_data.split(',')[1]  # Remove the 'data:image/png;base64,' part
#             img = Image.open(io.BytesIO(base64.b64decode(image_data)))
#             img.save('captured_face.png')  # Save the captured image

#             # Now, perform face detection using OpenCV
#             image = cv2.imread('captured_face.png')
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#             # Load the pre-trained face detector
#             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#             # Detect faces in the image
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             # If a face is detected, you can register the user
#             if len(faces) > 0:
#                 print("Face detected!")
#                 # Here you can add additional code for registering the user
#                 # For example, you might store the captured face data in a database or create a user profile
#             else:
#                 print("No face detected.")
#                 return jsonify({"message": "No face detected. Please try again."}), 400

#             return jsonify({"message": "Face registered successfully"}), 200

#     return render_template('add_user.html')



# if __name__ == '__main__':
#     app.run(debug=True)

#####home.html##


