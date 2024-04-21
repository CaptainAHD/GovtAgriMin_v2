import streamlit as st
import os
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from openai import OpenAI
import json
from google.cloud import texttospeech
from langdetect import detect
import base64
import requests
from google.oauth2 import service_account
from googletrans import Translator
from google_trans_new import google_translator  
translator = google_translator()  

st.set_page_config(layout="wide")

# Load the JSON credentials file directly
# with open("GOOGLE_APPLICATION_CREDENTIALS_JSON.json") as f:
#     service_account_info = json.load(f)
# key = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
# os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]=key
#  os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = (
#     "GOOGLE_APPLICATION_CREDENTIALS_JSON.json"
# )
# os.environ["OPENAI_API_KEY"] = ""
# API_KEY = ""
client = OpenAI()

audio_file_path = ""  # Define audio_file_path globally
credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
# Initialize Google Cloud Text-to-Speech client
text_to_speech_client = texttospeech.TextToSpeechClient(credentials=credentials)

@st.cache_resource
def create_retriever(top_k, source_language):
    index = load_index_from_storage(
        storage_context=StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="vector store"),
            vector_store=FaissVectorStore.from_persist_dir(persist_dir="vector store"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="vector store"),
        )
    )
    return index.as_retriever(retriever_mode="embedding", similarity_top_k=int(top_k)), source_language

def detect_language(text):
    try:
        if len(text.strip()) < 3:  # Check if text is too short
            # st.warning("Input text is too short for language detection.")
            return "en"  # Default to English
        language = detect(text)
        return language
    except Exception as e:
        st.error(f"Language detection failed: {e}")
        return "en"  # Default to English if language detection fails
    
def text_to_speech(text, audio_format=texttospeech.AudioEncoding.MP3):
    language_code = detect_language(text)
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=audio_format
    )

    response = text_to_speech_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content


def translate_to_english(text):
    try:
        translated_text = translator.translate(text, dest='en').text
        return translated_text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text  # Return the original text if translation fails

# def translate_to_english(text, target_language):
#     try:
#         response = client.translate(
#             text=text,
#             target_language=target_language,
#             model="gpt-4"
#         )
#         translated_text = response.choices[0].text.strip()
#         return translated_text
#     except Exception as e:
#         st.error(f"Translation failed: {e}")
#         return text  # Return the original text if translation fails

# def translate_to_english(text, source):
#     prompt = f"Translate the following text from {source} to English:\n\n{text}\n\nTranslate to English:"
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {
#                 "role": "system",
#                 "content": prompt
#             }
#         ],
#         temperature=0,
#         max_tokens=150,
#         stop=["\n"]
#     )
#     translated_text = response.choices[0].text.strip()
#     return translated_text

# Function to save audio data to a temporary file
def save_audio_to_tempfile(audio_data, samplerate):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
        sf.write(tmpfile_name, audio_data, samplerate)
    return tmpfile_name

# Function to transcribe audio using Deepgram API
def transcribe_audio(audio_file_path):
    api_key = "e07390d5c0b035bf435df507032ad66181f5eafa"
    url = "https://api.deepgram.com/v1/listen?model=nova-2"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    with open(audio_file_path, "rb") as f:
        audio_data = f.read()
    response = requests.post(url, headers=headers, data=audio_data)
    print("Response status code:", response.status_code)  # Debugging statement
    if response.status_code == 200:
        result = response.json()
        print("Transcription result:", result)  # Debugging statement
        if "results" in result and "channels" in result["results"] and result["results"]["channels"]:
            transcripts = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcripts
        else:
            print("No transcripts found in result.")  # Debugging statement
    else:
        print("Failed to transcribe audio. Error:", response.text)  # Debugging statement
    return None

# Function to save uploaded audio file
def save_uploaded_file(uploaded_file):
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_audio.wav"

transcribed_text = ""  # Define a default value for transcribed_text

# Add audio input functionality
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    # Save uploaded audio file
    file_path = save_uploaded_file(audio_file)
    # Transcribe audio and update query input field
    st.write("Transcribing audio...")
    transcribed_text = transcribe_audio(file_path)
    if transcribed_text:
        st.write("Transcription complete!")
    else:
        st.write("Failed to transcribe audio.")
query = st.text_input(label="Please enter your query - ", key="query_input")
# Modify query input field to allow for multiple languages
source_language = st.selectbox("Select Source Language:", ["English", "Spanish", "French", "German", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese", "Maithili"]) # Add more languages as needed
if source_language != "English":
    translated_query = translate_to_english(
        text= query
    )
else:
    translated_query = query

# Update query input field with transcribed text

top_k = st.number_input(label="Top k - ", min_value=3, max_value=5, value=3, key="top_k_input")
# Proceed with semantic search
retriever, source_language = create_retriever(top_k, source_language)
# Rest of your code for semantic search with the provided query

if query and top_k:
    col1, col2 = st.columns([3, 2])
    with col1:
        response = []
        for i in retriever.retrieve(translated_query):
            response.append(
                {
                    "Document": i.metadata["link"][40:-4],
                    "Source": i.metadata["link"],
                    "Text": i.get_text(),
                    "Score": i.get_score(),
                }
            )
        st.json(response)

    with col2:
        summary = st.empty()
        top3 = []
        top3_couplet = []
        top3_name = []
        for i in response:
            top3.append(i["Text"])
            top3_name.append(i["Document"])
        temp_summary = []
        # translated_query = translate_to_english(text=transcribed_text)
        # translated_query = translate_to_english(text=transcribed_text, target_language="en")
        for resp in client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"Act as a query answering GPT for The Ministry of Agriculture and Farmers Welfare, India. You answer queries of officers and farmers using your knowledgebase. Now answer the {query}, using the following knowledgebase:{top3} Your knowledgebase also contains name of the document, give it when answering so as to making your answer clear: {top3_name}. Strictly answer based on the available knowledge base. And remember, you must answer the query in easy to understand, everyday spoken language of {query}",
                },
                {
                    "role": "user",
                    "content": f"""Summarize the following interpretation of couplets in context of the {query}":

{top3_name[2]}
Summary:
{top3[2]}

{top3_name[1]}
Summary:
{top3[1]}

{top3_name[0]}
Summary:
{top3[0]}""",
                },
            ],
            stream=True,
        ):
            if resp.choices[0].finish_reason == "stop":
                break
            temp_summary.append(resp.choices[0].delta.content)
            result = "".join(temp_summary).strip()
            for phrase, link in {
                "Thrips": "https://drive.google.com/file/d/1Tnps02E_hBCgrdiS3etVV_J3hjT0xEyf/view?usp=share_link",
                "Whitefly": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
                "White Fly": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
                "whiteflies": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
                "PBW": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Pink Bollworm": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "pink bollworms": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Cotton PBW Larva": "https://drive.google.com/file/d/1l8HOlfZNbce_qHbaZujXO4KB_ug_SZZ3/view?usp=share_link",
                "Cotton Whitefly damage symptom": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "Cotton Whitefly damage symptoms": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "damage symptoms of Cotton Whitefly": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "Whitefly damage symptoms": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "Fall Army worm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
                "Fall Army Worm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
                "Fall Armyworm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
                "FF adult on Mango": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
                "FF damage to Indian crops": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
                "fruit flies on mangoes": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
                "FF Egg laying": "https://drive.google.com/file/d/1BVaNTtlG9Y7nSiOUqAS7yhfVnjDLPMkr/view?usp=share_link",
                "FF fruit damage": "https://drive.google.com/file/d/1oSRuO3M2D1wfiTPqA9VSxzSgambN7BXF/view?usp=share_link",
                "FF Larve damage": "https://drive.google.com/file/d/1Nr_ZwQEAIlgWoNjIEuXW_LG5yu_s7eHT/view?usp=share_link",
                "FF Oozing": "https://drive.google.com/file/d/1Sht1JZGlg_SqUWo0rN1stPL1FGqUYGtZ/view?usp=share_link",
                "FF Puncture": "https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link",
                "Fruit Fly": "https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link",
                "Fruitfly in Mango fruit": "https://drive.google.com/file/d/16zarIaupOIWAK2GrpBmQy214MqLfML53/view?usp=share_link",
                "Fruitfly in Mango leaf": "https://drive.google.com/file/d/1de4XhE1RQ5GKvOZcmkoz9Yi-q1JlQo1w/view?usp=share_link",
                "bore hole caused by larva of Yellow stem borer": "https://drive.google.com/file/d/1guo1cO2f1IjRPztTiZS9OemjFLq8KTiP/view?usp=share_link",
                "bore hole of YSB larva": "https://drive.google.com/file/d/1_k0msr8JRUp5uUUKh5dVBHepLNzb-Oyd/view?usp=share_link",
                "larva of Yellow stem borer": "https://drive.google.com/file/d/1L9WOrmqUPOUrzib17USsXrBWU6EcgYxX/view?usp=share_link",
                "Moth of Yellow Stem borer on paddy crop": "https://drive.google.com/file/d/12J37UHo_P5zPWAn4zU3zw35nDdzsIp1K/view?usp=share_link",
                "Moth of Yellow Stem borer": "https://drive.google.com/file/d/1St9fNNmMy1Sy_p_W6hTtjqhHF2UbGyfb/view?usp=share_link",
                "RICE- Yellow stem borer- Scirpophaga incertulas": "https://drive.google.com/file/d/1dw5hlAwPQFk5WodHbY72FkWwLDmdmCMr/view?usp=share_link",
                "Cotton PBW Damage Symptom": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "symptoms for Pink Bollworm (PBW) damage": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "symptoms of Cotton Whitefly damage": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Cotton PBW (Pink Bollworm) Damage Symptoms": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Cotton PBW (Pink Bollworm) Damage Symptom": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link"
                # ... (other phrase-link pairs)
            }.items():
                if phrase in result:
                    result = result.replace(phrase, f"[{phrase}]({link})")
            summary.markdown(result)
        # print(result)

        # Automatically speak the generated summary
        st.write("")
        st.write("")
        st.write("")

        st.write("Audio")
        audio_content = text_to_speech(result)
        audio_file_path = "data:audio/mp3;base64," + base64.b64encode(audio_content).decode("utf-8")
        st.audio(audio_file_path, format="audio/mp3")































# import streamlit as st
# import os
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.core import load_index_from_storage, StorageContext
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core.storage.index_store import SimpleIndexStore
# from openai import OpenAI
# import json
# from google.cloud import texttospeech
# from langdetect import detect
# import base64
# import requests
# from google.oauth2 import service_account

# st.set_page_config(layout="wide")

# # Load the JSON credentials file directly
# # with open("GOOGLE_APPLICATION_CREDENTIALS_JSON.json") as f:
# #     service_account_info = json.load(f)
# # key = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
# # os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]=key
# #  os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = (
# #     "GOOGLE_APPLICATION_CREDENTIALS_JSON.json"
# # )
# # os.environ["OPENAI_API_KEY"] = ""
# # API_KEY = ""
# client = OpenAI()

# audio_file_path = ""  # Define audio_file_path globally
# credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
# # Initialize Google Cloud Text-to-Speech client
# text_to_speech_client = texttospeech.TextToSpeechClient(credentials=credentials)

# @st.cache_resource
# def create_retriever(top_k, source_language):
#     index = load_index_from_storage(
#         storage_context=StorageContext.from_defaults(
#             docstore=SimpleDocumentStore.from_persist_dir(persist_dir="vector store"),
#             vector_store=FaissVectorStore.from_persist_dir(persist_dir="vector store"),
#             index_store=SimpleIndexStore.from_persist_dir(persist_dir="vector store"),
#         )
#     )
#     return index.as_retriever(retriever_mode="embedding", similarity_top_k=int(top_k)), source_language

# def detect_language(text):
#     try:
#         language = detect(text)
#         return language
#     except Exception as e:
#         st.error(f"Language detection failed: {e}")
#         return "en"  # Default to English if language detection fails
    
# def text_to_speech(text, audio_format=texttospeech.AudioEncoding.MP3):
#     language_code = detect_language(text)
#     synthesis_input = texttospeech.SynthesisInput(text=text)

#     voice = texttospeech.VoiceSelectionParams(
#         language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )

#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=audio_format
#     )

#     response = text_to_speech_client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )

#     return response.audio_content

# # Function to translate text to English
# def translate_to_english(text, source_language):
#     return text, source_language

# # Function to save audio data to a temporary file
# def save_audio_to_tempfile(audio_data, samplerate):
#     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
#         tmpfile_name = tmpfile.name
#         sf.write(tmpfile_name, audio_data, samplerate)
#     return tmpfile_name

# # Function to transcribe audio using Deepgram API
# def transcribe_audio(audio_file_path):
#     api_key = "e07390d5c0b035bf435df507032ad66181f5eafa"
#     url = "https://api.deepgram.com/v1/listen?model=nova-2"
#     headers = {
#         "Authorization": f"Token {api_key}",
#         "Content-Type": "audio/wav",
#     }
#     with open(audio_file_path, "rb") as f:
#         audio_data = f.read()
#     response = requests.post(url, headers=headers, data=audio_data)
#     print("Response status code:", response.status_code)  # Debugging statement
#     if response.status_code == 200:
#         result = response.json()
#         print("Transcription result:", result)  # Debugging statement
#         if "results" in result and "channels" in result["results"] and result["results"]["channels"]:
#             transcripts = result["results"]["channels"][0]["alternatives"][0]["transcript"]
#             return transcripts
#         else:
#             print("No transcripts found in result.")  # Debugging statement
#     else:
#         print("Failed to transcribe audio. Error:", response.text)  # Debugging statement
#     return None

# # Function to save uploaded audio file
# def save_uploaded_file(uploaded_file):
#     with open("temp_audio.wav", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return "temp_audio.wav"

# transcribed_text = ""  # Define a default value for transcribed_text

# # Add audio input functionality
# audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
# if audio_file is not None:
#     st.audio(audio_file, format="audio/wav")
#     # Save uploaded audio file
#     file_path = save_uploaded_file(audio_file)
#     # Transcribe audio and update query input field
#     st.write("Transcribing audio...")
#     transcribed_text = transcribe_audio(file_path)
#     if transcribed_text:
#         st.write("Transcription complete!")
#     else:
#         st.write("Failed to transcribe audio.")

# # Modify query input field to allow for multiple languages
# source_language = st.selectbox("Select Source Language:", ["English", "Spanish", "French", "German", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese", "Maithili"]) # Add more languages as needed
# if source_language != "English":
#     translated_query = transcribed_text
# else:
#     translated_query = transcribed_text

# # Update query input field with transcribed text
# query = st.text_input(label="Please enter your query - ", value=translated_query, key="query_input")
# top_k = st.number_input(label="Top k - ", min_value=3, max_value=5, value=3, key="top_k_input")
# # Proceed with semantic search
# retriever, source_language = create_retriever(top_k, source_language)
# # Rest of your code for semantic search with the provided query

# if query and top_k:
#     col1, col2 = st.columns([3, 2])
#     with col1:
#         response = []
#         for i in retriever.retrieve(query):
#             response.append(
#                 {
#                     "Document": i.metadata["link"][40:-4],
#                     "Source": i.metadata["link"],
#                     "Text": i.get_text(),
#                     "Score": i.get_score(),
#                 }
#             )
#         st.json(response)

#     with col2:
#         summary = st.empty()
#         top3 = []
#         top3_couplet = []
#         top3_name = []
#         for i in response:
#             top3.append(i["Text"])
#             top3_name.append(i["Document"])
#         temp_summary = []
#         translated_query = translate_to_english(
#             text=query, source_language=source_language
#         )  # Assuming English is the source language
#         for resp in client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"Act as a query answering GPT for The Ministry of Agriculture and Farmers Welfare, India. You answer queries of officers and farmers using your knowledgebase. Now answer the {translated_query}, using the following knowledgebase:{top3} Your knowledgebase also contains name of the document, give it when answering so as to making your answer clear: {top3_name}. Strictly answer based on the available knowledge base. And remember, you must answer the query in easy to understand, everyday spoken language of {query}",
#                 },
#                 {
#                     "role": "user",
#                     "content": f"""Summarize the following interpretation of couplets in context of the {query}":

# {top3_name[2]}
# Summary:
# {top3[2]}

# {top3_name[1]}
# Summary:
# {top3[1]}

# {top3_name[0]}
# Summary:
# {top3[0]}""",
#                 },
#             ],
#             stream=True,
#         ):
#             if resp.choices[0].finish_reason == "stop":
#                 break
#             temp_summary.append(resp.choices[0].delta.content)
#             result = "".join(temp_summary).strip()
#             for phrase, link in {
#                 "Thrips": "https://drive.google.com/file/d/1Tnps02E_hBCgrdiS3etVV_J3hjT0xEyf/view?usp=share_link",
#                 "Whitefly": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
#                 "White Fly": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
#                 "whiteflies": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
#                 "PBW": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "Pink Bollworm": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "pink bollworms": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "Cotton PBW Larva": "https://drive.google.com/file/d/1l8HOlfZNbce_qHbaZujXO4KB_ug_SZZ3/view?usp=share_link",
#                 "Cotton Whitefly damage symptom": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
#                 "Cotton Whitefly damage symptoms": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
#                 "damage symptoms of Cotton Whitefly": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
#                 "Whitefly damage symptoms": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
#                 "Fall Army worm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
#                 "Fall Army Worm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
#                 "Fall Armyworm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
#                 "FF adult on Mango": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
#                 "FF damage to Indian crops": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
#                 "fruit flies on mangoes": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
#                 "FF Egg laying": "https://drive.google.com/file/d/1BVaNTtlG9Y7nSiOUqAS7yhfVnjDLPMkr/view?usp=share_link",
#                 "FF fruit damage": "https://drive.google.com/file/d/1oSRuO3M2D1wfiTPqA9VSxzSgambN7BXF/view?usp=share_link",
#                 "FF Larve damage": "https://drive.google.com/file/d/1Nr_ZwQEAIlgWoNjIEuXW_LG5yu_s7eHT/view?usp=share_link",
#                 "FF Oozing": "https://drive.google.com/file/d/1Sht1JZGlg_SqUWo0rN1stPL1FGqUYGtZ/view?usp=share_link",
#                 "FF Puncture": "https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link",
#                 "Fruit Fly": "https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link",
#                 "Fruitfly in Mango fruit": "https://drive.google.com/file/d/16zarIaupOIWAK2GrpBmQy214MqLfML53/view?usp=share_link",
#                 "Fruitfly in Mango leaf": "https://drive.google.com/file/d/1de4XhE1RQ5GKvOZcmkoz9Yi-q1JlQo1w/view?usp=share_link",
#                 "bore hole caused by larva of Yellow stem borer": "https://drive.google.com/file/d/1guo1cO2f1IjRPztTiZS9OemjFLq8KTiP/view?usp=share_link",
#                 "bore hole of YSB larva": "https://drive.google.com/file/d/1_k0msr8JRUp5uUUKh5dVBHepLNzb-Oyd/view?usp=share_link",
#                 "larva of Yellow stem borer": "https://drive.google.com/file/d/1L9WOrmqUPOUrzib17USsXrBWU6EcgYxX/view?usp=share_link",
#                 "Moth of Yellow Stem borer on paddy crop": "https://drive.google.com/file/d/12J37UHo_P5zPWAn4zU3zw35nDdzsIp1K/view?usp=share_link",
#                 "Moth of Yellow Stem borer": "https://drive.google.com/file/d/1St9fNNmMy1Sy_p_W6hTtjqhHF2UbGyfb/view?usp=share_link",
#                 "RICE- Yellow stem borer- Scirpophaga incertulas": "https://drive.google.com/file/d/1dw5hlAwPQFk5WodHbY72FkWwLDmdmCMr/view?usp=share_link",
#                 "Cotton PBW Damage Symptom": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "symptoms for Pink Bollworm (PBW) damage": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "symptoms of Cotton Whitefly damage": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "Cotton PBW (Pink Bollworm) Damage Symptoms": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
#                 "Cotton PBW (Pink Bollworm) Damage Symptom": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link"
#                 # ... (other phrase-link pairs)
#             }.items():
#                 if phrase in result:
#                     result = result.replace(phrase, f"[{phrase}]({link})")
#             summary.markdown(result)
#         # print(result)

#         # Automatically speak the generated summary
#         st.write("")
#         st.write("")
#         st.write("")

#         st.write("Audio")
#         audio_content = text_to_speech(result)
#         audio_file_path = "temp_audio_col2.mp3"
#         with open(audio_file_path, "wb") as f:
#             f.write(audio_content)
#         st.audio(audio_file_path, format="audio/mp3")
