from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import cv2
from models import db, User, Attendance
from face_recog import recognize_and_mark

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face_attendance.db'
app.config['UPLOAD_FOLDER'] = 'static/photos'
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    photo = request.files['photo']
    filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    user = User(name=name, image_path=filename)
    db.session.add(user)
    db.session.commit()
    return redirect('/')

@app.route('/scan')
def scan():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        recognize_and_mark(frame)
        cv2.imshow("Press 'q' to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return redirect('/attendance')

@app.route('/attendance')
def attendance():
    records = db.session.query(User.name, Attendance.timestamp).join(Attendance, User.id == Attendance.user_id).all()
    return render_template('attendance.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
