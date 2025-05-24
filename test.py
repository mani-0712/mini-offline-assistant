import sys
import time
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import ollama
import pyttsx3
from PyQt5.QtCore import QThread, pyqtSignal, QCoreApplication, QTimer, QMutex

# --- Configuration ---
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 0.75)
ENERGY_THRESHOLD = 0.02
SILENCE_TIMEOUT = 1.5
OLLAMA_MODEL_NAME = "JennyModel"
TTS_RATE = 200
TTS_VOLUME = 1.0

is_speaking = False

# --- Audio Recorder Thread ---
class AudioRecorderThread(QThread):
    speech_started = pyqtSignal()
    speech_ended = pyqtSignal(np.ndarray)
    energy_updated = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.is_speaking_external = False
        self.buffer = []
        self.recording = False
        self.last_speech_time = time.time()

    def set_speaking_state(self, speaking):
        self.is_speaking_external = speaking

    def run(self):
        self.is_running = True
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=self.audio_callback):
            while self.is_running:
                self.process_audio()
                time.sleep(0.01)

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            self.audio_queue.put(indata.copy())
            energy = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
            self.energy_updated.emit(energy)

    def process_audio(self):
        try:
            block = self.audio_queue.get(timeout=0.1)
            flat_block = block.flatten()
            energy = np.sqrt(np.mean(flat_block.astype(np.float32) ** 2))

            if energy > ENERGY_THRESHOLD:
                if not self.recording:
                    self.speech_started.emit()
                    self.recording = True
                    self.buffer = []
                self.buffer.append(flat_block)
                self.last_speech_time = time.time()
            elif self.recording:
                if time.time() - self.last_speech_time > SILENCE_TIMEOUT:
                    self.recording = False
                    if self.buffer:
                        full_audio = np.concatenate(self.buffer).astype(np.float32)
                        self.speech_ended.emit(full_audio)
                        self.buffer = []
        except queue.Empty:
            pass

    def stop(self):
        self.is_running = False
        self.wait()

# --- Transcriber Thread ---
class TranscriberThread(QThread):
    transcription_ready = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.audio_data = None

    def set_audio_data(self, audio_data):
        self.audio_data = audio_data

    def run(self):
        if self.audio_data is not None:
            try:
                start_time = time.time()
                segments, _ = self.model.transcribe(self.audio_data, language="en", beam_size=2)
                print(f"[Transcriber] Time: {time.time() - start_time:.2f}s")
                text = " ".join(segment.text.strip() for segment in segments)
                self.transcription_ready.emit(text)
            except Exception as e:
                print(f"[Transcriber] Error: {e}")

# --- TTS Thread ---
class TTSThread(QThread):
    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', TTS_RATE)
        self.engine.setProperty('volume', TTS_VOLUME)
        if voices:  # ✅ FIXED: avoid index error
            self.engine.setProperty('voice', voices[0].id)  # ✅ FIXED: safe default
        self.text = None
        self.should_stop = False
        self.mutex = QMutex()

    def set_text(self, text):
        self.mutex.lock()  # ✅ FIXED: protect shared access
        self.text = text
        self.should_stop = False
        self.mutex.unlock()

    def stop_speaking(self):
        self.should_stop = True
        try:
            self.engine.stop()
        except Exception as e:
            print(f"[TTS] Error stopping: {e}")

    def run(self):
        self.mutex.lock()
        local_text = self.text
        self.mutex.unlock()
        if local_text and not self.should_stop:
            try:
                self.engine.say(local_text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Error: {e}")

# --- LLM Interaction Thread ---
class LLMInteractionThread(QThread):
    llm_response_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client = ollama.Client(host="http://localhost:11434")  # ✅ FIXED: Added host
        self.prompt = None

    def set_prompt(self, prompt):
        self.prompt = prompt

    def run(self):
        if self.prompt:
            try:
                full_prompt = f"\nUser: {self.prompt}\nAI:"
                response = self.client.generate(model=OLLAMA_MODEL_NAME, prompt=full_prompt)
                reply = response.response.strip()
                self.llm_response_ready.emit(reply)
            except Exception as e:
                print(f"[LLM] Error: {e}")

# --- Main Voice Assistant Logic ---
class VoiceAssistant:
    def __init__(self):
        self.model = WhisperModel("medium.en", device="cuda", compute_type="float16")
        self.audio_thread = AudioRecorderThread()
        self.audio_thread.speech_started.connect(lambda: print("[Audio] Speech detected..."))
        self.audio_thread.speech_ended.connect(self.process_speech)
        self.audio_thread.energy_updated.connect(self.check_interrupt)

        self.tts_thread = None
        self.transcriber = None  # ✅ FIXED: prevent undefined access
        self.llm = None  # ✅ FIXED: prevent undefined access
        self.is_listening = False

    def start(self):
        print("[Assistant] Listening started. Say something!")
        self.is_listening = True
        self.audio_thread.start()

    def stop(self):
        print("[Assistant] Stopping...")
        self.is_listening = False
        self.audio_thread.stop()
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop_speaking()
            self.tts_thread.wait()
        if self.transcriber and self.transcriber.isRunning():  # ✅ FIXED: cleanup transcriber
            self.transcriber.quit()
            self.transcriber.wait()
        if self.llm and self.llm.isRunning():  # ✅ FIXED: cleanup LLM thread
            self.llm.quit()
            self.llm.wait()

    def process_speech(self, audio_data):
        print("[Assistant] Transcribing...")
        self.transcriber = TranscriberThread(self.model)
        self.transcriber.set_audio_data(audio_data)
        self.transcriber.transcription_ready.connect(self.handle_transcription)
        self.transcriber.start()

    def handle_transcription(self, text):
        global is_speaking
        print(f"You: {text}")

        if any(cmd in text.lower() for cmd in ["stop", "pause", "wait"]):
            print("[Assistant] Stopping speech...")
            is_speaking = False
            return

        self.llm = LLMInteractionThread()
        self.llm.set_prompt(text)
        self.llm.llm_response_ready.connect(self.handle_response)
        self.llm.start()

    def handle_response(self, reply):
        global is_speaking
        print(f"AI: {reply}")
        is_speaking = True

        if "```" in reply:
            print("[Assistant] Skipping code speech. Saying acknowledgment...")
            reply = "Here is the code."

        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop_speaking()
            self.tts_thread.wait()

        self.tts_thread = TTSThread()
        self.tts_thread.set_text(reply)
        self.tts_thread.start()
        is_speaking = False

    def check_interrupt(self, energy):
        global is_speaking
        if is_speaking and energy > ENERGY_THRESHOLD * 3:
            print("[Assistant] Interrupting speech...")
            if self.tts_thread and self.tts_thread.isRunning():
                self.tts_thread.stop_speaking()
                self.tts_thread.wait()
            is_speaking = False

# --- Main Execution ---
if __name__ == "__main__":
    app = QCoreApplication(sys.argv)
    assistant = VoiceAssistant()
    assistant.start()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        assistant.stop()
        print("\n[System] Terminated by user.")
