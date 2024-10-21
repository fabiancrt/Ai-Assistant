Jarvis AI Assistant


Jarvis is a personal AI assistant project designed to help you interact with your computer using natural language commands. Built with a focus on machine learning and leveraging powerful libraries like PyTorch, Jarvis integrates speech recognition, natural language understanding, and text-to-speech functionalities to provide a seamless user experience.
Table of Contents

    Features
    Architecture
    Installation
        Prerequisites
        Setup
    Usage
        ASR Server
        ASR Client (Windows)
        TTS Server
    Environment Variables
    Dependencies
    Docker
    Contributing
    License

Features

    Wake Word Detection: Listens for the wake word "Jarvis" to activate.
    Automatic Speech Recognition (ASR): Converts spoken commands into text using Vosk.
    Natural Language Understanding: Processes commands to perform tasks like fetching weather, news, or executing system operations.
    Text-to-Speech (TTS): Responds with synthesized speech using Azure Cognitive Services.
    Memory Management: Maintains conversation history for context-aware interactions.
    Extensible Architecture: Easily add new functionalities and integrations.

Architecture


The Jarvis AI Assistant consists of several components working together:

    ASR Server (asr_server.py): Handles incoming text commands, processes them using machine learning models, and generates responses.
    ASR Client (asr_windows.py): Listens for the wake word, captures audio input, and communicates with the ASR Server.
    Info Retriever (info_retriever): Fetches and processes information from various sources like Wikipedia, News API, and SerpAPI.
    TTS Server (tts_server.py): Converts text responses into speech using Azure Text-to-Speech.
    Docker Container: Containerizes the TTS Server for easy deployment.

Installation
Prerequisites

    Python 3.10+
    WSL (Windows Subsystem for Linux) for running the ASR Server on Windows.
    Virtual Environment: Recommended to manage dependencies.
    Azure Cognitive Services Account: For Text-to-Speech capabilities.
    Hugging Face Account: For accessing language models.
    Vosk Model: Download a suitable Vosk model for ASR.

Setup

    Clone the Repository

    bash

git clone https://github.com/yourusername/jarvis-ai-assistant.git
cd jarvis-ai-assistant

Create and Activate Virtual Environment

bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies

bash

    pip install --upgrade pip
    pip install -r requirements.txt

    Download Vosk Model

    Download a Vosk model from Vosk Models and extract it to C:/Users/user/Desktop/Jarvis/vosk-model/vosk.

Usage
ASR Server

The ASR Server handles processing of text commands and generating responses.

    Configure Environment Variables

    Create a .env file in the project root with the following variables:

    env

HUGGINGFACE_HUB_TOKEN=your_huggingface_token
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SERVICE_REGION=your_azure_service_region
OPENWEATHER_API_KEY=your_openweather_api_key
NEWSAPI_KEY=your_newsapi_key
SERPAPI_API_KEY=your_serpapi_api_key

Run the ASR Server

bash

    python asr_server.py

    Note: Ensure you are using WSL and have activated the virtual environment.

ASR Client (Windows)

The ASR Client listens for the wake word and sends commands to the ASR Server.

    Configure Wake Word Detection

    Replace the access_key in asr_windows.py with your Picovoice Porcupine access key.

    Run the ASR Client

    Open Command Prompt, navigate to the project directory, activate the virtual environment, and run:

    cmd

    python asr_windows.py

TTS Server

The TTS Server converts text responses into speech using Azure Cognitive Services.

    Configure Azure TTS

    Update tts_server.py with your Azure Speech Key and Region:

    python

AZURE_SPEECH_KEY = "YOUR_AZURE_SPEECH_KEY"
AZURE_SERVICE_REGION = "YOUR_SERVICE_REGION"

Run the TTS Server

bash

    python tts_server.py

Environment Variables

Ensure the following environment variables are set in your .env file:

    HUGGINGFACE_HUB_TOKEN: Token for accessing Hugging Face models.
    AZURE_SPEECH_KEY: Azure Cognitive Services Speech API key.
    AZURE_SERVICE_REGION: Azure Cognitive Services region.
    OPENWEATHER_API_KEY: API key for OpenWeather.
    NEWSAPI_KEY: API key for NewsAPI.
    SERPAPI_API_KEY: API key for SerpAPI.

Dependencies

All dependencies are listed in requirements.txt. Key libraries include:

    Machine Learning & NLP: torch, transformers, accelerate
    ASR & TTS: vosk, pyaudio, azure-cognitiveservices-speech
    Web Framework: fastapi, uvicorn
    Utilities: aiohttp, beautifulsoup4, requests, numpy, pydub

Docker

A Dockerfile is provided to containerize the TTS Server.

    Build the Docker Image

    bash

docker build -t jarvis-tts-server .

Run the Docker Container

bash

    docker run -d -p 50051:50051 --name jarvis-tts jarvis-tts-server

    The container exposes port 50051 for TTS requests and includes a health check endpoint.

Contributing

Contributions are welcome! Please follow these steps:

    Fork the Repository

    Create a Feature Branch

    bash

git checkout -b feature/YourFeature

Commit Your Changes

bash

git commit -m "Add your feature"

Push to the Branch

bash

    git push origin feature/YourFeature

    Create a Pull Request

License

This project is licensed under the MIT License.

Developed with ❤️ by Fabian.
Acknowledgements

    Vosk
    Azure Cognitive Services
    Hugging Face
    Picovoice Porcupine
    SerpAPI
    NewsAPI

