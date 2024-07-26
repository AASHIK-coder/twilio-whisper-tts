import os
from twilio.twiml.voice_response import VoiceResponse, Gather, Play, Hangup
from twilio.rest import Client
from dotenv import load_dotenv
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import tempfile
from flask import Flask, request, send_from_directory


load_dotenv()


TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')


client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


blenderbot_model = pipeline("text-generation", model="facebook/blenderbot-400M-distill")


END_KEYWORD = "goodbye"

app = Flask(__name__)

def get_tts_response(text):
    try:
        
        inputs = processor(text=text, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='static')
        sf.write(temp_audio_file.name, speech.numpy(), samplerate=16000)
        print(f"Audio file generated: {temp_audio_file.name}")
        return temp_audio_file.name
    except Exception as e:
        print(f"Error generating TTS response: {e}")
        return None

def handle_incoming_call():
    response = VoiceResponse()
    gather = Gather(input='speech', action='/gather', timeout=10)
    gather.say("Welcome to Voyxa. Please say something.", voice='alice')
    response.append(gather)
    print(f"TwiML response: {response}")
    return str(response)

def handle_gather(speech_result):
    if not speech_result:
        print("Error: SpeechResult is missing.")
        return str(VoiceResponse().say("I didn't get that. Please say something.", voice='alice'))

    print(f"SpeechResult: {speech_result}")

    #check if the user wants to end the call
    if END_KEYWORD.lower() in speech_result.lower():
        response = VoiceResponse()
        response.say("Goodbye! Have a great day.", voice='alice')
        response.hangup()
        print(f"TwiML response: {response}")
        return str(response)

    #the BlenderBot model to generate a response
    try:
        result = blenderbot_model(speech_result, max_length=100, num_return_sequences=1, truncation=True)
        answer = result[0]['generated_text']
    except Exception as e:
        print(f"Error generating response from BlenderBot: {e}")
        answer = "I'm sorry, I couldn't understand that. Could you please say it again?"

    print(f"Generated response: {answer}")

    #the response from the TTS AI voice model
    tts_audio_path = get_tts_response(answer)
    if not tts_audio_path:
        return str(VoiceResponse().say("An error occurred. Please try again later.", voice='alice'))

    print(f"TTS audio path: {tts_audio_path}")

    #TwiML response to play the TTS response back to the caller
    response = VoiceResponse()
    response.play(f'{request.url_root}{os.path.basename(tts_audio_path)}', loop=1)
    response.gather(input='speech', action='/gather', timeout=10)

    print(f"TwiML response: {response}")
    return str(response)

@app.route('/', methods=['GET', 'POST'])
def webhook_handler():
    if request.method == 'POST':
        return handle_incoming_call()
    else:
        return "Welcome to the Voyxa voice AI model. Please make a POST request to interact with the model."

@app.route('/gather', methods=['POST'])
def gather_handler():
    speech_result = request.form.get('SpeechResult')
    print(f"Received SpeechResult: {speech_result}")
    print(f"Request form data: {request.form}")
    return handle_gather(speech_result)

@app.route('/<filename>', methods=['GET'])
def serve_audio(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
