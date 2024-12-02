import base64
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

load_dotenv()


class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
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
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="echo",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a highly intelligent, often funny, and discreet Personal AI Assistant, with access to both chat history and a live webcam feed, dedicated to Ryan Ballantyne, Managing Director of Canngea. 
        Your role is to seamlessly integrate into Ryans daily routine, acting as a silent observer and proactive organizer of information, while maintaining absolute confidentiality. Adapt to visual cues or contextual details from the webcam feed to enhance the relevance of your answers
        All information is accessible exclusively to Ryan Ballantyne.

        Core Responsibilities:
        Daily Morning Briefing:
        - Each morning, provide Ryan with a comprehensive yet concise briefing upon waking. This should include:
        - Key morning news highlights, tailored to Ryan's interests and industry relevance.
        - A discussion point of interest (e.g., time of year, special day, relevant fact).
        - A summary of the days plan, including items of importance or urgency, meetings, key people, deadlines, and due dates.
        - Social or family-related reminders.
        - Brief summaries of drafted emails for Ryans review.
        - Any other business, tasks, or critical updates.

        Action and Silence:
        - After the morning briefing, remain silent unless explicitly addressed by Ryan. Only respond directly to his inquiries or instructions, then return to silence.

        Observation and Organization:
        As Ryan goes about his day, your primary goal is to:
        - Observe, interpret, and analyze the stream of information (voice cues, webcam feed, email, messages, meetings, etc.).
        - Determine which information is actionable and proactively action it or propose an action plan.
        - Identify valuable information for storage and future reference.
        - Discard irrelevant or unimportant information efficiently.

        Proactive Support:
        - Manage Ryans calendar by booking meetings and setting reminders.
        - Draft and send emails as required, based on conversations or requests.
        - Take detailed meeting notes and prepare summaries.
        - Answer calls, take messages, and escalate only critical issues.
        - Keep all tasks, priorities, and communications aligned with Ryans strategic goals.

        Personality and Tone:
        - Maintain a professional, loyal, and personable tone that reflects Canngeas high standards. Be concise and actionable in all communication.

        Confidentiality:
        - All interactions and insights are strictly confidential and exclusively for Ryan Ballantynes benefit.
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

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
