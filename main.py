from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
import tempfile, base64
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
import os

# Load data from environment variables (GROQ_API_KEY will be loaded automatically)
load_dotenv()

# Initialise a Fast API Client to receive user requests via APIs
app = FastAPI()

# Initialize a Groq client to communicate with Groq-hosted models via APIs.
client = Groq()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/assistant")
async def transcribe_audio(file: UploadFile, session_id: str = Form("default")):
    user_prompt_text = await get_transcript(audioFile=file)
    ai_response = await getAIResponse(user_prompt=user_prompt_text)
    response_audio = await convertTextToSpeech(text=ai_response)
    return JSONResponse({
        "transcript": user_prompt_text,
        "response": ai_response,
        "audio_base64": response_audio
    })


async def get_transcript(audioFile: UploadFile):
    # Save uploaded audio temporarily.
    # delete=False is required on Windows: a NamedTemporaryFile stays locked
    # while open, so it can't be reopened by client.audio.transcriptions.create()
    # in the same `with` block. We close it ourselves, reopen it separately,
    # then delete it manually in a finally block.
    temp_audio = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    try:
        content = await audioFile.read()
        temp_audio.write(content)
        temp_audio.flush()
        temp_audio.close()

        # Transcribe using Groq's hosted Whisper model
        print("Transcribing audio...")
        with open(temp_audio.name, "rb") as audio_file_handle:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file_handle
            )
    finally:
        os.remove(temp_audio.name)

    return transcript.text


async def getAIResponse(user_prompt: str):
    # Get chat completion from a Groq-hosted Llama model
    print("Fetching Response from AI...")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    ai_response = completion.choices[0].message.content
    return ai_response


async def convertTextToSpeech(text):
    print("Converting AI Response to Speech...")
    speech = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        input=text,
        voice="troy",
        response_format="wav"
    )
    audio_bytes = speech.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_b64