import base64
import os

from threading import Lock, Thread

import cv2
import openai

from cv2 import VideoCapture, imencode

from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.schema.messages import SystemMessage

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

from pyaudio import PyAudio, paInt16

from speech_recognition import Microphone, Recognizer, UnknownValueError

from flask import Flask, request, jsonify

from werkzeug.utils import secure_filename

import sqlite3

import json


load_dotenv()


# Database setup for user profiles and preferences

def init_db():

    conn = sqlite3.connect('assistant.db')

    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS user_preferences (

                 user_id TEXT PRIMARY KEY,

                 preferences TEXT

                 )''')

    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (

                 user_id TEXT,

                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

                 message TEXT

                 )''')
    conn.commit()
    conn.close()


init_db()


class WebcamStream:

    def __init__(self, camera_index=0):

        self.stream = VideoCapture(index=camera_index)

        if not self.stream.isOpened():

            raise Exception("Could not open video device")

        _, self.frame = self.stream.read()

        self.running = False

        self.lock = Lock()


    def start(self):

        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self


    def update(self):

        while self.running:

            ret, frame = self.stream.read()

            if not ret:
                continue

            with self.lock:

                self.frame = frame


    def read(self, encode=False):

        with self.lock:

            frame = self.frame.copy()

        if encode:

            _, buffer = imencode(".jpeg", frame)

            return base64.b64encode(buffer)
        return frame


    def stop(self):

        self.running = False

        if self.thread.is_alive():

            self.thread.join()
        self.stream.release()


class Assistant:

    def __init__(self, model, user_id="default"):

        self.chain = self._create_inference_chain(model)

        self.user_id = user_id

        self.chat_message_history = ChatMessageHistory()
        self.load_user_preferences()


    def load_user_preferences(self):

        conn = sqlite3.connect('assistant.db')

        c = conn.cursor()

        c.execute("SELECT preferences FROM user_preferences WHERE user_id=?", (self.user_id,))

        row = c.fetchone()

        if row:

            self.preferences = json.loads(row[0])

        else:

            self.preferences = {"voice": "default", "language": "english"}

            c.execute("INSERT INTO user_preferences (user_id, preferences) VALUES (?, ?)", 

                      (self.user_id, json.dumps(self.preferences)))
            conn.commit()
        conn.close()


    def save_user_preferences(self):

        conn = sqlite3.connect('assistant.db')

        c = conn.cursor()

        c.execute("REPLACE INTO user_preferences (user_id, preferences) VALUES (?, ?)", 

                  (self.user_id, json.dumps(self.preferences)))
        conn.commit()
        conn.close()


    def answer(self, prompt, image):

        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(

            {"prompt": prompt, "image_base64": image.decode()},

            config={"configurable": {"session_id": self.user_id}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

            self.save_chat_history(prompt, response)


    def _tts(self, response):

        # Use user preference for voice

        voice = self.preferences.get("voice", "default")

        # Placeholder for TTS implementation

        # Example using pyttsx3 (synchronous, replace with async or streaming as needed)

        import pyttsx3

        engine = pyttsx3.init()
        jls_extract_var = 'voice'
        engine.setProperty(jls_extract_var, voice)

        engine.say(response)

        engine.runAndWait()


    def _create_inference_chain(self, model):

        SYSTEM_PROMPT = """

        You are a personalized assistant that uses chat history and the provided image to answer questions.

        Your job is to assist the user effectively and efficiently.


        Use clear and concise language. Be friendly, helpful, and show personality.
        """


        prompt_template = ChatPromptTemplate.from_messages(

            [

                SystemMessage(content=SYSTEM_PROMPT),

                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",

                    [

                        {"type": "text", "text": "{prompt}"},

                        {

                            "type": "image_url",

                            "image_url": "data:image/jpeg;base64,{image_base64}",

                        },

                    ],
                ),

            ]
        )


        chain = prompt_template | model | StrOutputParser()


        return RunnableWithMessageHistory(
            chain,

            lambda _: self.chat_message_history,

            input_messages_key="prompt",

            history_messages_key="chat_history",
        )


    def save_chat_history(self, user_message, assistant_response):

        conn = sqlite3.connect('assistant.db')

        c = conn.cursor()

        c.execute("INSERT INTO chat_history (user_id, message) VALUES (?, ?)", 

                  (self.user_id, f"User: {user_message}"))

        c.execute("INSERT INTO chat_history (user_id, message) VALUES (?, ?)", 

                  (self.user_id, f"Assistant: {assistant_response}"))
        conn.commit()
        conn.close()


    def update_preferences(self, new_preferences):

        self.preferences.update(new_preferences)

        self.save_user_preferences()


# Initialize webcam stream

webcam_stream = WebcamStream().start()


# Initialize model

model = ChatOpenAI(model="gpt-4o")


# Initialize assistant with a unique user ID

assistant = Assistant(model, user_id="user123")


# Flask app for handling user interactions and preferences

app = Flask(__name__)


@app.route('/update_preferences', methods=['POST'])

def update_preferences():

    data = request.json
    assistant.update_preferences(data)

    return jsonify({"status": "success", "preferences": assistant.preferences})


@app.route('/get_chat_history', methods=['GET'])

def get_chat_history():

    conn = sqlite3.connect('assistant.db')

    c = conn.cursor()

    c.execute("SELECT timestamp, message FROM chat_history WHERE user_id=? ORDER BY timestamp", 
              (assistant.user_id,))

    rows = c.fetchall()
    conn.close()

    return jsonify({"chat_history": rows})


def audio_callback(recognizer, audio):

    try:

        prompt = recognizer.recognize_whisper(audio, model="base", language=assistant.preferences.get("language", "english"))

        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:

        print("There was an error processing the audio.")


# Setup recognizer and microphone

recognizer = Recognizer()

microphone = Microphone()

with microphone as source:

    recognizer.adjust_for_ambient_noise(source)


stop_listening = recognizer.listen_in_background(microphone, audio_callback)


# Run Flask app in a separate thread

def run_flask():

    app.run(port=5000)


flask_thread = Thread(target=run_flask)

flask_thread.start()


# Main loop for displaying webcam feed

try:

    while True:

        frame = webcam_stream.read()

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) in [27, ord("q")]:

            break

except KeyboardInterrupt:
    pass

finally:

    webcam_stream.stop()

    cv2.destroyAllWindows()

    stop_listening(wait_for_stop=False)

    # Optionally, terminate Flask thread if needed

