import time
import speech_recognition as sr
import pyttsx3
from threading import Thread


class SpeechRecognizer:
    def __init__(self, timeout=3, phrase_time_limit=5):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.result = None
        self.thread = None
        self.listening = False
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

    def _recognize_speech(self, delay):
        time.sleep(delay)
        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                print("Listening timed out")
                self.result = None
                self.listening = False
                return
        try:
            # Recognize speech using Google Web Speech API
            response = self.recognizer.recognize_google(audio)
            print(f"Recognized: {response}")
            self.result = response.lower()
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            self.result = None
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}"
            )
            self.result = None
        finally:
            self.listening = False

    def listen(self, *, delay=0.5):
        if self.listening:
            print("Already listening...")
            return
        self.listening = True
        self.thread = Thread(target=self._recognize_speech, args=(delay,))
        self.thread.start()

    def user_accepted(self):
        if self.result is not None:
            return self.result == "yes"
        return None

    def get_result(self):
        return self.result

    def is_listening(self):
        return self.listening

    def close(self):
        if self.thread is not None:
            self.thread.join()
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
