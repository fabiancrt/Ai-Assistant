# Jarvis AI Assistant

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)

## Overview

Jarvis is an AI Assistant developed as a personal project to deepen understanding of machine learning and natural language processing. Built with PyTorch, Jarvis leverages various technologies to provide functionalities such as speech recognition, text-to-speech, information retrieval, and more. This project serves as a comprehensive learning tool for enthusiasts looking to explore the capabilities of AI assistants.

## Features

- **Automatic Speech Recognition (ASR):** Converts spoken language into text using Vosk.
- **Natural Language Processing (NLP):** Processes and understands commands using Hugging Face Transformers.
- **Text-to-Speech (TTS):** Synthesizes speech from text using Azure Cognitive Services.
- **Information Retrieval:** Fetches information from Wikipedia, news APIs, and performs web searches.
- **Custom Memory Management:** Remembers user interactions for contextually relevant responses.
- **Cross-Platform Compatibility:** Runs on both Windows and WSL environments.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Install Apex](#4-install-apex)
  - [5. Install Vosk](#5-install-vosk)
  - [6. Download Required Models](#6-download-required-models)
    - [a. Vosk Models](#a-vosk-models)
    - [b. TTS Models](#b-tts-models)
  - [7. Set Up Environment Variables](#7-set-up-environment-variables)
- [Usage](#usage)
  - [Running ASR Server (WSL)](#running-asr-server-wsl)
  - [Running Windows Listener](#running-windows-listener)
- [File Structure](#file-structure)
- [Important Notes](#important-notes)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Requirements

Before setting up Jarvis, ensure you have the following installed on your system:

- **Python 3.10:** [Download Python](https://www.python.org/downloads/)
- **WSL (Windows Subsystem for Linux):** [Install WSL](https://docs.microsoft.com/en-us/windows/wsl/install)
- **Docker (Optional):** [Get Docker](https://www.docker.com/get-started)
- **Virtualenv:** For creating isolated Python environments. Install via pip:

    ```bash
    pip install virtualenv
    ```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/jarvis-ai-assistant.git
cd jarvis-ai-assistant
```
2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

bash

python -m venv venv

Activate the virtual environment:

On Windows:

bash

venv\Scripts\activate

On WSL/Linux:

bash

source venv/bin/activate

3. Install Dependencies

Install the required Python packages using pip:

bash

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Note: The apex package is excluded from requirements.txt due to its large size and specific installation requirements. Follow the instructions below to install Apex manually.
4. Install Apex

Apex is a PyTorch extension for mixed precision and distributed training. It must be installed manually.

Installation Steps:

Clone the Apex Repository:

bash

git clone https://github.com/NVIDIA/apex.git
cd apex

Install Apex:

bash

pip install -v --disable-pip-version-check --no-cache-dir ./

Note: Ensure that your CUDA version is compatible with the Apex version you are installing. Refer to the Apex GitHub for detailed installation instructions and requirements.
5. Install Vosk

Vosk is used for offline speech recognition.

bash

pip install vosk==0.3.45

6. Download Required Models

Jarvis requires specific models for speech recognition and text-to-speech functionalities. Due to their large sizes, these models are not included in the repository.
a. Vosk Models

Download Link: Vosk Models (replace with actual link)

Setup:

    Download the desired Vosk model (e.g., vosk-model-small-en-us-0.15.zip).

    Extract the contents.

    Place the extracted folder inside the vosk directory of the project:

    Jarvis-AI-Assistant/
    ├── vosk/
    │   └── vosk-model-small-en-us-0.15/

b. TTS Models

Jarvis uses tacotron2.nemo and waveglow.nemo for text-to-speech.

Download Links:

    Tacotron2 Nemo Model (replace with actual link)
    WaveGlow Nemo Model (replace with actual link)

Setup:

    Download both .nemo files.

    Place them in the following directory:

    makefile

    C:\Users\user\Desktop\Jarvis\docker_tts\models\

Note: Adjust the path according to your system's directory structure if necessary.
7. Set Up Environment Variables

Create a .env file in the root directory of the project and add the following variables:

env

HUGGINGFACE_HUB_TOKEN=your_huggingface_token
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SERVICE_REGION=your_azure_service_region
OPENWEATHER_API_KEY=your_openweather_api_key
NEWSAPI_KEY=your_newsapi_key
SERPAPI_API_KEY=your_serpapi_api_key

Replace the placeholder values with your actual API keys.
Usage
Running ASR Server (WSL)

The ASR server handles automatic speech recognition and runs on WSL.

    Activate the Virtual Environment:

    bash

source venv/bin/activate

Navigate to the Project Directory:

bash

cd path/to/jarvis-ai-assistant

Run the ASR Server:

bash

    python asr_server.py

Note: Ensure you are using WSL and have activated the virtual environment before running the server.
Running Windows Listener

The Windows listener listens for the wake word and sends commands to the ASR server.

    Open Command Prompt:

    Navigate to the directory containing asr_windows.py:

    cmd

cd C:\Users\user\Desktop\Jarvis\asr_windows

Run the Windows Listener:

cmd

    python asr_windows.py

The listener will wait for the wake word ("jarvis") and process subsequent voice commands.
File Structure

plaintext

Jarvis-AI-Assistant/
├── asr_server.py
├── asr_windows.py
├── info_retriever/
│   └── info_retriever.py
├── docker_tts/
│   ├── Dockerfile
│   ├── tts_server.py
│   └── models/
│       ├── tacotron2.nemo
│       └── waveglow.nemo
├── requirements.txt
├── .env
├── apex/ (Not included, install manually)
├── vosk/ (Not included, install and add models manually)
└── README.md

Important Notes

    Apex Folder: The apex folder is excluded from the repository due to its large size. Users must install Apex manually by following the instructions in the Apex GitHub Repository.

    Vosk Folder: The vosk folder containing the speech recognition models is not included. Users need to download the appropriate Vosk models and place them in the vosk directory as outlined in the Installation section.

    Virtual Environment: It is highly recommended to create and activate a virtual environment before installing dependencies to avoid conflicts with other projects.

    Model Downloads: Both tacotron2.nemo and waveglow.nemo are large files and are not included in the repository. Users must download these models from the provided links and place them in the specified directory.

Contributing

Contributions are welcome! Whether it's improving documentation, fixing bugs, or adding new features, your input is valuable. Please follow these steps to contribute:

    Fork the Repository

    Create a New Branch

    bash

git checkout -b feature/YourFeature

Commit Your Changes

bash

git commit -m "Add Your Feature"

Push to the Branch

bash

    git push origin feature/YourFeature

    Open a Pull Request

License

This project is licensed under a special License.
Acknowledgements

    PyTorch
    Transformers by Hugging Face
    Vosk
    Azure Cognitive Services
    SerpAPI
    NewsAPI
    Apex by NVIDIA
    Wikipedia API
    BeautifulSoup
    FastAPI
    Docker
