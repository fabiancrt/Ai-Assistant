

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import azure.cognitiveservices.speech as speechsdk
import io
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Azure_TTS_Server")


executor = ThreadPoolExecutor(max_workers=4)


AZURE_SPEECH_KEY = "YOUR_AZURE_SPEECH_KEY"  
AZURE_SERVICE_REGION = "YOUR_SERVICE_REGION"  

speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

def synthesize_audio(text):
    try:
        logger.debug(f"Synthesizing audio for text: {text}")
        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Audio synthesized successfully.")
            audio_stream = io.BytesIO(result.audio_data)
            return audio_stream
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {cancellation_details.error_details}")
            raise HTTPException(status_code=500, detail="Speech synthesis canceled.")
    except Exception as e:
        logger.error(f"Error during synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Synthesis failed.")

def iterfile(buf):
    chunk_size = 1024  
    while True:
        chunk = buf.read(chunk_size)
        if not chunk:
            break
        yield chunk

@app.post('/synthesize')
async def synthesize(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.warning("Invalid content type. Expected application/json.")
        return JSONResponse(content={"error": "Invalid content type. Expected application/json."}, status_code=400)

    text = data.get('text', '').strip()

    if not text:
        logger.warning("No text provided for synthesis.")
        return JSONResponse(content={"error": "No text provided for synthesis."}, status_code=400)

    logger.info(f"Received synthesis request for text: {text}")

    try:
        
        buf = await asyncio.get_event_loop().run_in_executor(executor, synthesize_audio, text)
        buf.seek(0)

        
        return StreamingResponse(iterfile(buf), media_type='audio/mpeg', headers={'Content-Disposition': 'attachment; filename="response.mp3"'})

    except HTTPException as he:
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        logger.error(f"Error during synthesis: {e}", exc_info=True)
        return JSONResponse(content={"error": "Synthesis failed."}, status_code=500)

@app.get('/health')
async def health_check():
    return JSONResponse(content={"status": "OK"}, status_code=200)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=50051)
