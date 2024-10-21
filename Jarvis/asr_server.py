# asr_server.py

import socket
import threading
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
import requests
import re
from memory import (
    initialize_db,
    set_persistent,
    get_persistent,
    add_short_term,
    get_short_term,
    cleanup_short_term,
    start_memory_cleanup
)
from info_retriever.info_retriever import InfoRetriever  # Ensure correct import
import logging
from dotenv import load_dotenv
import warnings
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import io
from datetime import datetime  # Import datetime for date functionality

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='/c/Users/user/Desktop/Jarvis/asr_server.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG for detailed logs
)
logger = logging.getLogger("ASR_Server")

# Suppress specific warnings
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Initialize memory
initialize_db()
start_memory_cleanup()

# Token for Hugging Face
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not token:
    logging.critical("HUGGINGFACE_HUB_TOKEN is not set. Please set it in your .env file.")
    exit(1)
else:
    logging.info("HUGGINGFACE_HUB_TOKEN is set.")

# Initialize Accelerator
accelerator = Accelerator()

# Define the model name
model_name = "lmsys/vicuna-7b-v1.5"  # Updated model identifier

# Initialize tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side='right',
        truncation_side='right',
        use_auth_token=token,
        model_max_length=2048,
        padding=True
    )
except Exception as e:
    logging.critical(f"Failed to load tokenizer: {e}", exc_info=True)
    raise e

# Assign eos_token as pad_token if pad_token is not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("Set pad_token to eos_token.")

# Load the model in float16 without bitsandbytes
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatically maps layers to available devices
        offload_folder="/c/Users/user/Desktop/Jarvis/offload",
        low_cpu_mem_usage=True,
        use_auth_token=token,
        use_safetensors=False
    )

    logging.info("Vicuna 7B model loaded successfully.")

    # Prepare the model with Accelerator
    model = accelerator.prepare(model)
    logging.info("Model prepared with Accelerator.")

    # Update generation configuration to enable sampling
    model.config.update({
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_length": 512
    })
    logging.debug("Model generation configuration updated.")

except Exception as e:
    logging.critical(f"Failed to load AI model: {e}", exc_info=True)
    raise e

# Initialize InfoRetriever with tokenizer and model
info_retriever = InfoRetriever(tokenizer, model)

# Example: Set persistent memory
set_persistent("user_name", "Fabian")
set_persistent("assistant_name", "Jarvis")
set_persistent("relationship", "Owner")

# Azure Speech Service Configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION")  # e.g., "eastus"

# Debug logs to verify environment variables
logger.debug(f"AZURE_SPEECH_KEY: {AZURE_SPEECH_KEY}")
logger.debug(f"AZURE_SERVICE_REGION: {AZURE_SERVICE_REGION}")

if not AZURE_SPEECH_KEY or not AZURE_SERVICE_REGION:
    logging.critical("Azure Speech Service credentials are not set in environment variables.")
    raise Exception("Azure Speech Service credentials are missing.")

speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

def synthesize_audio_azure(text):
    try:
        logger.debug(f"Synthesizing audio for text: {text}")

        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Azure Speech Service synthesized the audio successfully.")

            # Access the audio data as bytes
            audio_data = result.audio_data

            # Convert MP3 bytes to WAV using pydub
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))  # <-- Uses io
            buf = io.BytesIO()
            audio_segment.export(buf, format="wav")
            buf.seek(0)

            logger.info("Audio converted to WAV format successfully.")
            return buf

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Azure Speech Service synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {cancellation_details.error_details}")
            raise Exception("Azure Speech Service synthesis canceled.")

    except Exception as e:
        logger.error(f"Error during Azure synthesis: {e}", exc_info=True)
        raise e

def handle_client_connection(client_socket):
    try:
        data = client_socket.recv(4096)
        if not data:
            logging.warning("No data received from client.")
            return
        text = data.decode('utf-8').strip()
        logging.info(f"Received ASR text: {text}")
        response_text = process_command(text)
        logging.info(f"Generated response: {response_text}")
        audio_buf = synthesize_audio_azure(response_text)
        if audio_buf:
            audio_data = audio_buf.read()
            audio_size = len(audio_data)
            client_socket.sendall(audio_size.to_bytes(4, byteorder='big'))
            client_socket.sendall(audio_data)
            logging.info(f"Audio data of size {audio_size} bytes sent to client.")
        else:
            logging.error("Failed to synthesize audio.")
    except Exception as e:
        logging.error(f"Error handling client connection: {e}", exc_info=True)
    finally:
        client_socket.close()
        logging.debug("Client socket closed.")

def process_command(command):
    try:
        # Retrieve persistent memory
        user_name = get_persistent("user_name")
        assistant_name = get_persistent("assistant_name")
        relationship = get_persistent("relationship")

        # Handle 'system_greet' command separately
        if command.lower() == "system_greet":
            response = f"Hello there, {user_name}! How may I help you?"
            add_short_term("conversation", command, response)
            logging.info(f"Generated response for 'system_greet': {response}")
            return response

        # Retrieve short-term memory
        short_term = get_short_term()

        # Perform information retrieval if needed
        retrieved_info = info_retriever.retrieve_information(command)
        logging.debug(f"Retrieved information: {retrieved_info}")

        # Ensure retrieved_info is a string
        if retrieved_info is None:
            logger.warning(f"Retrieved information is None for command '{command}'.")
            retrieved_info = "I'm sorry, I couldn't find any information related to your request."

        # Determine if retrieved_info is an error or valid information
        info_section = ""
        error_phrases = ["an error occurred", "please specify", "no search results found"]

        if not any(substring in retrieved_info.lower() for substring in error_phrases):
            info_section = f"Here is some information I found:\n{retrieved_info}\n\n"
        else:
            # If retrieval failed, provide a fallback message
            info_section = f"Here is some information I could find:\n{retrieved_info}\n\n"

        # Get current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build prompt with memory, retrieved information, and current date
        prompt = (
            f"You are {assistant_name}, a highly intelligent and helpful personal assistant.\n"
            f"Your owner is {user_name}, and your relationship is {relationship}.\n"
            f"Current date and time: {current_datetime}\n"
            "You remember past interactions to provide contextually relevant responses.\n\n"
            f"{info_section}"
            "Conversation History:\n"
        )

        for entry in short_term:
            category, cmd, resp, timestamp = entry
            if category == "conversation":
                prompt += f"User: {cmd}\n{assistant_name}: {resp}\n"

        # Add user command without additional instructions
        prompt += f"User: {command}\n{assistant_name}: "

        logging.debug(f"Final prompt sent to AI model:\n{prompt}")

        # Tokenize the input prompt with attention mask
        encoding = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Adjust based on model's capacity
        )

        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)

        # Generate response using AI model
        logging.info("Generating response using the AI model.")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,  # Adjust as needed
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).split(f"{assistant_name}:")[-1].strip()
        logging.info(f"AI model response: {response}")

        # Sanitize response
        sanitized_response = sanitize_response(response)
        logging.debug(f"Sanitized AI model response: {sanitized_response}")

        # Add to short-term memory
        add_short_term("conversation", command, sanitized_response)
        logging.debug("Added response to short-term memory.")

        # Limit response length to prevent TTS cutoff
        if len(sanitized_response) > 500:  # Adjust the limit as per TTS capabilities
            sanitized_response = sanitized_response[:497] + "..."
            logging.warning("Response truncated to prevent TTS cutoff.")

        # Free up GPU memory
        del input_ids, attention_mask, output_ids
        torch.cuda.empty_cache()

        return sanitized_response
    except Exception as e:
        logging.error(f"Error processing command '{command}': {e}", exc_info=True)
        return "I'm sorry, I encountered an error while processing your request."


def sanitize_response(response):
    """
    Remove any unwanted instructions, tokens, or emojis from the AI model's response.
    """
    # Remove any text after specific unwanted phrases
    unwanted_phrases = [
        "Please provide the actual text",
        "Is there anything else I can assist you with?",
        "Please let me know if you need further assistance.",
        "Based on your owner's profile",  # Remove profile-based instructions
        "Please provide a concise and accurate response based on the information provided."  # Added phrase
    ]
    for phrase in unwanted_phrases:
        if phrase in response:
            response = response.split(phrase)[0].strip()
    
    # Remove emojis using regex
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # Emoticons
                           u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                           u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                           u"\U0001F1E0-\U0001F1FF"  # Flags
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    response = emoji_pattern.sub(r'', response)
    
    return response

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', 65432))  # Update if using a different port
    server_socket.listen(5)
    logging.info("Jarvis ASR Server is running and waiting for connections...")

    try:
        while True:
            client_sock, address = server_socket.accept()
            logging.info(f"Accepted connection from {address}")
            client_handler = threading.Thread(
                target=handle_client_connection,
                args=(client_sock,)
            )
            client_handler.start()
    except KeyboardInterrupt:
        logging.info("Shutting down server.")
    finally:
        server_socket.close()
        logging.debug("Server socket closed.")

if __name__ == "__main__":
    main()
