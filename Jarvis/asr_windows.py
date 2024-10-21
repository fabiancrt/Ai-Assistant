import vosk
import pyaudio
import json
import socket
import pvporcupine
import numpy as np
import logging
import time
import sys
import wave
import io
import winsound
import os

greet_sent = False 

def setup_logging():
    logging.basicConfig(
        filename='asr_windows.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG 
    )

def detect_wake_word():
    access_key = ""  # Replace with your Porcupine access key
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=["jarvis"]
        )
        logging.debug("Porcupine wake word engine initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Porcupine: {e}", exc_info=True)
        print(f"Error initializing Porcupine: {e}")
        return False

    pa = pyaudio.PyAudio()

    try:
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        logging.debug("Audio stream for wake word detection opened successfully.")
    except Exception as e:
        logging.error(f"Error opening audio stream: {e}", exc_info=True)
        print(f"Error opening audio stream: {e}")
        porcupine.delete()
        pa.terminate()
        return False

    print("Listening for wake word...")
    logging.info("Listening for wake word...")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                logging.info("Wake word detected!")
                return True  
    except Exception as e:
        logging.error(f"Error during wake word detection: {e}", exc_info=True)
        print(f"Error during wake word detection: {e}")
        return False
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()
        logging.debug("Wake word detection stream closed and Porcupine deleted.")

def send_text_to_server(text, server_ip='localhost', port=65432):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((server_ip, port))
            sock.sendall(text.encode('utf-8'))
            logging.info(f"Sent text to ASR server: {text}")
            
            audio_size_data = sock.recv(4)
            if not audio_size_data:
                logging.error("No audio size data received from ASR server.")
                print("No audio size data received from ASR server.")
                return
            
            audio_size = int.from_bytes(audio_size_data, byteorder='big')
            logging.info(f"Expecting audio data of size: {audio_size} bytes")
            
            audio_data = b''
            while len(audio_data) < audio_size:
                packet = sock.recv(4096)
                if not packet:
                    break
                audio_data += packet
            
            if len(audio_data) != audio_size:
                logging.warning(f"Expected audio size {audio_size} bytes, but received {len(audio_data)} bytes.")
                print(f"Expected audio size {audio_size} bytes, but received {len(audio_data)} bytes.")
            
            with open("response.wav", "wb") as f:
                f.write(audio_data)
                logging.info("Audio data saved to response.wav for testing.")
            
            play_audio(audio_data)
            logging.info("Audio data received and played successfully.")
            
    except ConnectionRefusedError:
        logging.error("Unable to connect to ASR server. Ensure it is running.", exc_info=True)
        print("Unable to connect to ASR server. Ensure it is running.")
    except Exception as e:
        logging.error(f"Error sending/receiving data to/from ASR server: {e}", exc_info=True)
        print(f"Error sending/receiving data to/from ASR server: {e}")

def play_audio(audio_bytes):
    try:
        logging.debug(f"Preparing to play audio of {len(audio_bytes)} bytes.")
        
        temp_wav_path = "temp_response.wav"
        
        with open(temp_wav_path, "wb") as f:
            f.write(audio_bytes)
            logging.debug("Audio data written to temp_response.wav")
        
        logging.debug("Playing audio with winsound.")
        winsound.PlaySound(temp_wav_path, winsound.SND_FILENAME)
        logging.info("Audio playback completed.")
        
        os.remove(temp_wav_path)
        logging.debug("Temporary WAV file removed.")
        
        time.sleep(1)  
    except Exception as e:
        logging.error(f"Error playing audio with winsound: {e}", exc_info=True)
        print(f"Error playing audio with winsound: {e}")

def main():
    global greet_sent
    setup_logging()
    logging.debug("Starting asr_windows.py main function.")
    while True:
        recognizer = None  
        try:
            logging.debug("Waiting for wake word.")
            wake_word_detected = detect_wake_word()
            if not wake_word_detected:
                logging.warning("Wake word detection failed or interrupted.")
                continue

            logging.info("Wake word detected! Initializing Vosk model.")

            try:
                model = vosk.Model("C:/Users/user/Desktop/Jarvis/vosk-model/vosk")
                logging.info("Vosk model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading Vosk model: {e}", exc_info=True)
                continue

            recognizer = vosk.KaldiRecognizer(model, 16000)
            audio_interface = pyaudio.PyAudio()
            try:
                stream = audio_interface.open(format=pyaudio.paInt16,
                                              channels=1,
                                              rate=16000,
                                              input=True,
                                              frames_per_buffer=8000)
                stream.start_stream()
                logging.info("Audio stream started. Listening for commands.")
            except Exception as e:
                logging.error(f"Error opening audio stream: {e}", exc_info=True)
                continue


            if not greet_sent:
                logging.debug("Sending 'system_greet' to ASR server.")
                send_text_to_server("system_greet", 'localhost', 65432)
                greet_sent = True
                logging.info("'system_greet' sent successfully.")

            while True:
                try:
                    data = stream.read(4000, exception_on_overflow=False)
                    logging.debug("Read 4000 bytes from audio stream.")
                except Exception as e:
                    logging.error(f"Error reading from audio stream: {e}", exc_info=True)
                    break

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    logging.debug(f"Recognizer result: {result}")
                    text = result.get('text', "")
                    if text:
                        print(f"Recognized: {text}")
                        logging.info(f"Recognized: {text}")
                        if "jarvis stop" in text.lower():
                            print("Deactivating...")
                            logging.info("Deactivating...")
                            greet_sent = False 
                            break  
                        elif text.lower() == "system_greet":
                            logging.warning("Detected 'system_greet' in commands. Ignoring.")
                            continue
                        else:
                            send_text_to_server(text, 'localhost', 65432)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Shutting down gracefully.")
            continue  
        except Exception as e:
            logging.error(f"Unhandled exception in main loop: {e}", exc_info=True)
            print(f"Unhandled exception: {e}")
            greet_sent = False
            continue
        finally:
            if recognizer is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                    audio_interface.terminate()
                    logging.debug("Audio stream terminated.")
                except Exception as e:
                    logging.error(f"Error during cleanup: {e}", exc_info=True)
            else:
                logging.debug("Recognizer was not initialized; skipping stream termination.")
            del recognizer
            logging.info("ASR Windows script loop terminated.")

if __name__ == "__main__":
    main()
