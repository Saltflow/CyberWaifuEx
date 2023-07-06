#! python3.7
# Originate: github.com:davabase/whisper_real_time.git

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

class StreamWhisper():

    def __init__(self, model="medium", non_english=False, energy_threshold=1000,
                record_timeout=2, phrase_timeout=3) -> None:

      # Thread safe Queue for passing data from the threaded recording callback.
      self.data_queue = Queue()
      # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
      self.recorder = sr.Recognizer()
      self.recorder.energy_threshold = energy_threshold
      # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
      self.recorder.dynamic_energy_threshold = False


      self.source = sr.Microphone(sample_rate=16000)

      if model != "large" and not non_english:
          model = model + ".en"
      self.audio_model = whisper.load_model(model)

      self.record_timeout = record_timeout
      self.phrase_timeout = phrase_timeout

      self.temp_file = NamedTemporaryFile().name
      self.transcription = ['']
      
      self.recorder.adjust_for_ambient_noise(self.source)

      def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        self.data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
      # We could do this manually but SpeechRecognizer provides a nice helper.
      self.recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)


    def getData(self):
      # Cue the user that we're ready to go.
      print("Model loaded.\n")
      # The last time a recording was retreived from the queue.
      phrase_time = None
      last_sample = bytes()
      while True:
          try:
              now = datetime.utcnow()
              # Pull raw recorded audio from the queue.
              if not self.data_queue.empty():
                  phrase_complete = False
                  # If enough time has passed between recordings, consider the phrase complete.
                  # Clear the current working audio buffer to start over with the new data.
                  if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
                      last_sample = bytes()
                      phrase_complete = True
                  # This is the last time we received new audio data from the queue.
                  phrase_time = now

                  # Concatenate our current audio data with the latest audio data.
                  while not self.data_queue.empty():
                      data = self.data_queue.get()
                      last_sample += data

                  # Use AudioData to convert the raw data to wav data.
                  audio_data = sr.AudioData(last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                  wav_data = io.BytesIO(audio_data.get_wav_data())

                  # Write wav data to the temporary file as bytes.
                  with open(self.temp_file, 'w+b') as f:
                      f.write(wav_data.read())

                  # Read the transcription.
                  result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                  text = result['text'].strip()

                  # If we detected a pause between recordings, add a new item to our transcripion.
                  # Otherwise edit the existing one.
                  if phrase_complete:
                      self.transcription.append(text)
                  else:
                      self.transcription[-1] = text

                  # Clear the console to reprint the updated transcription.
                  os.system('cls' if os.name=='nt' else 'clear')
                  for line in self.transcription:
                      print(line)
                  # Flush stdout.
                  print('', end='', flush=True)

                  # Infinite loops are bad for processors, must sleep.
                  sleep(0.25)
          except KeyboardInterrupt:
              break

      print("\n\nTranscription:")
      for line in self.transcription:
          print(line)

if __name__ == "__main__":
    i_whisper = StreamWhisper()
    i_whisper.getData()