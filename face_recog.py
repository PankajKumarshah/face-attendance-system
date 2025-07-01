import face_recognition
import cv2
import os
from datetime import datetime
from models import User, Attendance, db

def load_known_faces():
    known_encodings = []
    known_ids = []
    for user in User.query.all():
        path = os.path.join('static/photos', user.image_path)
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_ids.append(user.id)
    return known_encodings, known_ids

def recognize_and_mark(frame):
    known_encodings, known_ids = load_known_faces()
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for face_encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        dist = face_recognition.face_distance(known_encodings, face_encoding)
        if len(dist) > 0:
            best_match = dist.argmin()
            if matches[best_match]:
                user_id = known_ids[best_match]
                existing = Attendance.query.filter_by(user_id=user_id).first()
                if not existing:
                    db.session.add(Attendance(user_id=user_id, timestamp=datetime.now()))
                    db.session.commit()
