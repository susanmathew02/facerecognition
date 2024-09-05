import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known face images and their encodings
known_face_encodings = []
known_face_names = []

jobs_image = face_recognition.load_image_file("jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
known_face_encodings.append(jobs_encoding)
known_face_names.append("jobs")

ratan_tata_image = face_recognition.load_image_file("tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]
known_face_encodings.append(ratan_tata_encoding)
known_face_names.append("ratan tata")

mona_image = face_recognition.load_image_file("mona.jpg")
mona_encoding = face_recognition.face_encodings(mona_image)[0]
known_face_encodings.append(mona_encoding)
known_face_names.append("mona")

students = known_face_names.copy()

# Open video capture
video_capture = cv2.VideoCapture(0)

# Open CSV file for writing attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_filename = current_date + '.csv'
csv_file = open(csv_filename, 'w+', newline='')
csv_writer = csv.writer(csv_file)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR color (used by OpenCV) to RGB color (used by face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # If the recognized face is a student, mark attendance
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                csv_writer.writerow([name, current_time])

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()