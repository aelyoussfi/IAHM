import sys
from textwrap import fill
from turtle import bgcolor, color
import cv2
import threading
import tkinter as tk
import tkinter.ttk as ttk
from queue import Queue
from PIL import Image
from PIL import ImageTk
#from aux import submit
from pathlib import Path
import numpy as np 
import keras
from keras.models import *
import tensorflow as tf
#from modelsSet import * 
#from dataSet import generateData, perc 
#from modelIHM import training
import os 
from tkinter_custom_button import TkinterCustomButton
from tkinter.messagebox import showinfo
import time 
#from train import brightness
import shutil
from face_recognition.face_recognition import face_recognition

class App(tk.Frame):
    def __init__(self, parent, title):
        tk.Frame.__init__(self, parent)
        self.is_running = False
        self.thread = None
        self.queue = Queue()
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (800, 600), "silver"))
        parent.wm_withdraw()
        parent.wm_title(title)
        self.create_ui() 
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        parent.wm_protocol("WM_DELETE_WINDOW", self.on_destroy)
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)
        parent.wm_deiconify() 
        

    def create_ui(self):
        bgColor = 'Black'
        self.button_frame = tk.Frame(self,background=bgColor,width=700,height=700)
        #self.button_frame.geometry("350x300")

        #test of custom
        #1. camera starting
        self.start_button = TkinterCustomButton(bg_color=bgColor,
                                            border_color="#BB8FCE",
                                            border_width=4,
                                            fg_color="#6C3483",
                                            hover_color="#A569BD",
                                            text_font=("Times New Roman", 12, "bold"),
                                            text="Cam ON",
                                            text_color="white",
                                            corner_radius=20,
                                            width=120,
                                            height=40,
                                            hover=True, command=self.start)
        self.start_button.place(x=30, y=615)

        #2.Stop button
        self.start_button = TkinterCustomButton(bg_color=bgColor,
                                            border_color="#BB8FCE",
                                            border_width=4,
                                            fg_color="#6C3483",
                                            hover_color="#A569BD",
                                            text_font=("Times New Roman", 12, "bold"),
                                            text="Freeze all ",
                                            text_color="white",
                                            corner_radius=20,
                                            width=120,
                                            height=40,
                                            hover=True, command=self.stop)
        self.start_button.place(x=320, y=615)

        #3. start inference
        self.start_button = TkinterCustomButton(bg_color=bgColor,
                                            border_color="#BB8FCE",
                                            border_width=4,
                                            fg_color="#6C3483",
                                            hover_color="#A569BD",
                                            text_font=("Times New Roman", 12, "bold"),
                                            text="Identify ",
                                            text_color="white",
                                            corner_radius=20,
                                            width=120,
                                            height=40,
                                            hover=True, command = self.start_inference)
        self.start_button.place(x=180, y=615)





        """
        self.text = tk.Label(self.button_frame,text = '')
        self.text.pack()
        """
    
        self.view = tk.Label(self, image=self.photo)
        self.view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
####################################################################################################
    def on_destroy(self):
        self.stop()
        self.after(10)
        if self.thread is not None:
            self.thread.join(0.2)
        self.winfo_toplevel().destroy()
####################################################################################################
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
####################################################################################################
    def start_inference(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.inference, args=())
        self.thread.daemon = True
        self.thread.start()
####################################################################################################
    
    def stop(self):
        self.is_running = False
####################################################################################################
#boite de dialogue
    def submit(self):  # Callback function for SUBMIT Button.
        self.text = self.text_field.get(1.0,"end-1c")  # For line 1, col 0 to end.
        name = self.text
        Path('C:\\Users\\Azeddine\\Documents\\IA\\smartHome\\metropole\\TrainPhotos\\'+name).mkdir(parents=True, exist_ok=True)
        self.currentPath = 'C:\\Users\\Azeddine\\Documents\\IA\\smartHome\\metropole\\TrainPhotos\\'+name
    
    def submit_verif(self):  # Callback function for SUBMIT Button.
        self.text1 = self.text_field1.get(1.0,"end-1c")  # For line 1, col 0 to end.
        name = self.text1
        Path('C:\\Users\\Azeddine\\Documents\\IA\\smartHome\\metropole\\verification\\'+name).mkdir(parents=True, exist_ok=True)
        self.currentPath = 'C:\\Users\\Azeddine\\Documents\\IA\\smartHome\\metropole\\verification\\'+name 
####################################################################################################
    def videoLoop(self):
        No=0
        cap = cv2.VideoCapture(No)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        while self.is_running:
            ret, frame = cap.read()
            #image = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            image = frame[:, :, ::-1]
            self.queue.put(image)
            self.event_generate('<<MessageGenerated>>')

####################################################################################################
    def inference(self):
        video_capture = cv2.VideoCapture(0)

        # Load a sample picture and learn how to recognize it.
        azeddine_image = face_recognition.load_image_file("azeddine.jpeg")
        azeddine_face_encoding = face_recognition.face_encodings(azeddine_image)[0]

        # Load a second sample picture and learn how to recognize it.
        yoann_image = face_recognition.load_image_file("yoann.jpg")
        yoann_face_encoding = face_recognition.face_encodings(yoann_image)[0]

        # Load a third sample picture and learn how to recognize it.
        mehdi_image = face_recognition.load_image_file("mehdi.jpg")
        mehdi_face_encoding = face_recognition.face_encodings(mehdi_image)[0]

        # Load a fourth sample picture and learn how to recognize it.
        honorat_image = face_recognition.load_image_file("honorat.png")
        honorat_face_encoding = face_recognition.face_encodings(honorat_image)[0]

        # Load a fifth sample picture and learn how to recognize it.
        benoit_image = face_recognition.load_image_file("benoit.png")
        benoit_face_encoding = face_recognition.face_encodings(benoit_image)[0]

        # Load a fifth sample picture and learn how to recognize it.
        Eric_image = face_recognition.load_image_file("Eric.jpg")
        Eric_face_encoding = face_recognition.face_encodings(Eric_image)[0]

        # Load a fifth sample picture and learn how to recognize it.
        PJ_image = face_recognition.load_image_file("Pierre-jean.jpg")
        PJ_face_encoding = face_recognition.face_encodings(PJ_image)[0]


        # Create arrays of known face encodings and their names
        known_face_encodings = [
            azeddine_face_encoding,
            yoann_face_encoding,
            mehdi_face_encoding,
            honorat_face_encoding,
            benoit_face_encoding,
            Eric_face_encoding,
            PJ_face_encoding

        ]
        known_face_names = [
            "azeddine",
            "yoann",
            "mehdi",
            "honorat",
            "benoit",
            "Eric",
            "Pierre-J."
        ]

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            self.queue.put(frame)
            self.event_generate('<<MessageGenerated>>')
            

####################################################################################################
    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            image = self.queue.get()
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image)
            self.view.configure(image=self.photo)
    

def main(args):
    root = tk.Tk()
    app = App(root, "OpenCV Image Viewer")
    root.mainloop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))