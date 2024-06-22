import speech_recognition as sr
import pyttsx3
from threading import Thread, Event


class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening_thread = None
        self.text = ""
        self.stop_listening_flag = Event()
        self.listening = False
        self.response = False

    def listen_once(self):
        def callback(recognizer, audio):
            try:
                self.text = recognizer.recognize_google(audio)
                self.response = True
                print(f"Recognized text: {self.text}")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
            finally:
                self.stop_listening_flag.set()

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.listening = True
        stop_listening = self.recognizer.listen_in_background(self.microphone, callback)

        # Keep the thread alive until stop_listening_flag is set
        while not self.stop_listening_flag.is_set():
            continue
        print("Trying to stop listening...")
        stop_listening()
        self.listening = False

    def start_listening(self):
        print("Listening...")
        self.response = False
        if not self.listening:
            self.stop_listening_flag.clear()
            self.listening_thread = Thread(target=self.listen_once)
            self.listening_thread.start()

    def stop_listening(self):
        print("Stopped listening")
        if self.listening_thread and self.listening:
            self.stop_listening_flag.set()
            self.listening_thread.join()
            self.listening_thread = None
            self.listening = False

    def user_accepted(self) -> bool:
        if self.text.lower() == "yes":
            return True
        elif self.text.lower() == "no":
            return False
        else:
            return False

    def get_text(self):
        return self.text

    def close(self):
        self.stop_listening()
        self.recognizer = None
        self.microphone = None


class TTS:
    def __init__(self, text):
        self.tts = pyttsx3.init()
        rate = self.tts.getProperty("rate")
        self.tts.setProperty("rate", rate - 80)
        thread = Thread(target=self.speak, args=(text,))
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    def speak(self, text):
        self.tts.say(text)
        self.tts.runAndWait()
