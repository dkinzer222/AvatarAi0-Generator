from google.cloud import speech_v1
from google.cloud import dialogflow
from google.cloud import texttospeech

def process_speech(audio_content):
    client = speech_v1.SpeechClient()
    
    audio = speech_v1.RecognitionAudio(content=audio_content)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US"
    )
    
    response = client.recognize(config=config, audio=audio)
    
    # Process with Dialogflow
    if response.results:
        text = response.results[0].alternatives[0].transcript
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path("my-project-101-436505", "user123")
        
        text_input = dialogflow.TextInput(text=text, language_code="en-US")
        query_input = dialogflow.QueryInput(text=text_input)
        
        dialogflow_response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )
        
        return {
            'text': dialogflow_response.query_result.fulfillment_text,
            'intent': dialogflow_response.query_result.intent.display_name
        }
    
    return {'text': 'Sorry, I did not understand that.', 'intent': None}
