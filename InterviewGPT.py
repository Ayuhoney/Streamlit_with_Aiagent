########################################## GROQ MODEL

# import pickle
# from pathlib import Path
# import websockets
# import base64
# import asyncio
# import json
# from groq import AsyncGroq
# import pyaudio
# import pandas as pd
# import plotly.express as px
# import streamlit as st
# from collections import deque
# import spacy
# from gtts import gTTS
# import tempfile
# import streamlit.components.v1 as components

# def play_tts(text):
#     try:
#         tts = gTTS(text=text, lang="en")
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
#             tts.save(tmpfile.name)
#             st.audio(tmpfile.name, format="audio/mp3", autoplay=True)
#     except Exception as e:
#         print(f"Error in TTS: {e}")
#         st.error("Error generating speech")


# st.set_page_config(page_title="InterviewGPT", page_icon=":brain:", layout="wide")

# # Hardcoded user credentials for simplicity
# credentials = {
#     "p": "p",
#     "vaddi": "password2"
# }

# def check_password(username, password):
#     return credentials.get(username) == password

# # Authentication form
# st.sidebar.title("Login")
# username = st.sidebar.text_input("Username")
# password = st.sidebar.text_input("Password", type="password")

# if st.sidebar.button("Login"):
#     if check_password(username, password):
#         st.session_state["authenticated"] = True
#         st.session_state["username"] = username
#     else:
#         st.sidebar.error("Invalid username or password")

# if "authenticated" not in st.session_state:
#     st.session_state["authenticated"] = False

# if st.session_state["authenticated"]:
    
#     # load_premium_ui()
#     # ---- SIDEBAR ----
#     if st.sidebar.button("Logout"):
#         st.session_state["authenticated"] = False
#         st.session_state["username"] = ""

#     st.sidebar.title(f"Welcome {st.session_state['username'].upper()}")

#     # ---- MAINPAGE ----
#     st.title(":brain: InterviewGPT")
#     st.markdown("##")

#     # Groq API
#     client = AsyncGroq(
#         api_key="",  
#     )
#     # AssemblyAI API key
#     auth_key = ""  
#     if "text" not in st.session_state:
#         st.session_state["text"] = "Listening..."
#         st.session_state["run"] = False

#     FRAMES_PER_BUFFER = 8000
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 16000
#     p = pyaudio.PyAudio()

#     # Starts recording
#     stream = p.open(
#         format=FORMAT,
#         channels=CHANNELS,
#         rate=RATE,
#         input=True,
#         frames_per_buffer=FRAMES_PER_BUFFER,
#     )

#     conversation_history = deque(maxlen=5)
#     transcript = []  # Global transcript

#     def start_listening():
#         st.session_state["run"] = True

#     def stop_listening():
#         with open("conversation.txt", "w") as file:
#             file.write("\n".join(transcript))
#         st.session_state["run"] = False

#     def apply_differential_privacy():
#         try:
#             nlp = spacy.load("en_core_web_sm")

#             with open("conversation.txt", "r") as file:
#                 lines = file.readlines()

#             user_lines = [
#                 line[len("User:"):].strip() for line in lines if line.startswith("User:")
#             ]
#             user_text = "\n".join(user_lines)

#             doc = nlp(user_text)
#             for ent in doc.ents:
#                 if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "PHONE"]:
#                     user_text = user_text.replace(ent.text, "[REDACTED]")

#             with open("conversation_redacted.txt", "w") as file:
#                 file.write(user_text)
#         except Exception as e:
#             print(f"Error in applying differential privacy: {e}")

#     start, stop = st.columns(2)
#     start.button("Start listening", on_click=start_listening)

#     stop.button(
#         "Stop listening",
#         on_click=lambda: [stop_listening(), apply_differential_privacy()],
#     )

#     URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

#     async def send_receive():
#         headers = {
#             "Authorization": auth_key
#         }

#         print(f"Connecting websocket to url {URL}")

#         try:
#             async with websockets.connect(URL, additional_headers=headers) as _ws:
#                 await asyncio.sleep(0.1)
#                 print("Receiving SessionBegins ...")

#                 session_begins = await _ws.recv()
#                 print(session_begins)
#                 print("Sending messages ...")

#                 async def send():
#                     try:
#                         while st.session_state["run"]:
#                             try:
#                                 data = stream.read(FRAMES_PER_BUFFER)
#                                 data = base64.b64encode(data).decode("utf-8")

#                                 json_data = json.dumps({"audio_data": str(data)})
#                                 await _ws.send(json_data)
#                                 await asyncio.sleep(0.01)

#                             except websockets.exceptions.ConnectionClosedError as e:
#                                 print(f"Send connection closed: {e}")
#                                 break
#                             except Exception as e:
#                                 print(f"Send error: {e}")
#                                 await asyncio.sleep(1)
#                     except Exception as e:
#                         print(f"Send function error: {e}")

#                 async def receive():
#                     try:
#                         while st.session_state["run"]:
#                             try:
#                                 result_str = await _ws.recv()
#                                 result_json = json.loads(result_str)
#                                 result = result_json.get("text", "")
                                
#                                 if result_json.get("message_type") == "FinalTranscript":
#                                     print(result)

#                                     st.session_state["text"] = f"<span style='color: orange;'>User:</span> {result}"
#                                     st.markdown(st.session_state["text"], unsafe_allow_html=True)
#                                     transcript.append(f"User: {result}")
#                                     conversation_history.append({"role": "user", "content": result})

#                                     if result:
#                                         messages = [
#                                             {"role": "system", "content": "You are a helpful assistant."}
#                                         ] + list(conversation_history)

#                                         try:
#                                             chat_completion = await client.chat.completions.create(
#                                                 messages=messages,
#                                                 model="llama3-70b-8192",
#                                                 temperature=0.5,
#                                                 max_tokens=300,
#                                                 stream=False
#                                             )
#                                             reply = chat_completion.choices[0].message.content
#                                             print(f"InterviewGPT: {reply}")
#                                             play_tts(reply)
#                                             conversation_history.append({"role": "assistant", "content": reply})
#                                             transcript.append(f"InterviewGPT: {reply}")

#                                             st.session_state["chatText"] = f"<span style='color: green;'>InterviewGPT:</span> {reply}"
#                                             st.markdown(st.session_state["chatText"], unsafe_allow_html=True)
#                                         except Exception as e:
#                                             print(f"Error in chat completion: {e}")
#                                             st.error("Error getting response from InterviewGPT")

#                             except websockets.exceptions.ConnectionClosedError as e:
#                                 print(f"Receive connection closed: {e}")
#                                 break
#                             except Exception as e:
#                                 print(f"Receive error: {e}")
#                                 await asyncio.sleep(1)
#                     except Exception as e:
#                         print(f"Receive function error: {e}")

#                 await asyncio.gather(send(), receive())
#         except Exception as e:
#             print(f"WebSocket connection error: {e}")
#             st.error("Error connecting to speech recognition service")

#     if st.session_state["run"]:
#         try:
#             asyncio.run(send_receive())
#         except Exception as e:
#             print(f"Main loop error: {e}")
#             st.error("An error occurred in the main application loop")



########################################### OPENAI- Turbo 3.5

import pickle
from pathlib import Path
import websockets
import base64
import asyncio
import json
import openai  # Import the OpenAI package
import pyaudio
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import deque
import spacy
from gtts import gTTS
import tempfile
import streamlit.components.v1 as components

def play_tts(text):
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            st.audio(tmpfile.name, format="audio/mp3", autoplay=True)
    except Exception as e:
        print(f"Error in TTS: {e}")
        st.error("Error generating speech")

st.set_page_config(page_title="InterviewGPT", page_icon=":brain:", layout="wide")

# Hardcoded user credentials for simplicity
credentials = {
    "admin": "admin"
}

def check_password(username, password):
    return credentials.get(username) == password

# Authentication form
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if check_password(username, password):
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
    else:
        st.sidebar.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:

    # ---- SIDEBAR ----
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""

    st.sidebar.title(f"Welcome {st.session_state['username'].upper()}")

    # ---- MAINPAGE ----
    st.title(":brain: Mock Interview BY Ai ")
    st.markdown("##")

    # OpenAI API key
    openai.api_key = ""
    auth_key = ""  

    if "text" not in st.session_state:
        st.session_state["text"] = "Listening..."
        st.session_state["run"] = False

    FRAMES_PER_BUFFER = 8000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()

    # Starts recording
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    conversation_history = deque(maxlen=5)
    transcript = []  # Global transcript

    def start_listening():
        st.session_state["run"] = True

    def stop_listening():
        with open("conversation.txt", "w") as file:
            file.write("\n".join(transcript))
        st.session_state["run"] = False

    def apply_differential_privacy():
        try:
            nlp = spacy.load("en_core_web_sm")

            with open("conversation.txt", "r") as file:
                lines = file.readlines()

            user_lines = [
                line[len("User:"):].strip() for line in lines if line.startswith("User:")
            ]
            user_text = "\n".join(user_lines)

            doc = nlp(user_text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "PHONE"]:
                    user_text = user_text.replace(ent.text, "[REDACTED]")

            with open("conversation_redacted.txt", "w") as file:
                file.write(user_text)
        except Exception as e:
            print(f"Error in applying differential privacy: {e}")

    start, stop = st.columns(2)
    start.button("Start listening", on_click=start_listening)

    stop.button(
        "Stop listening",
        on_click=lambda: [stop_listening(), apply_differential_privacy()],
    )

    URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

    async def send_receive():
        headers = {
            "Authorization": auth_key
        }

        print(f"Connecting websocket to url {URL}")

        try:
            async with websockets.connect(URL, additional_headers=headers) as _ws:
                await asyncio.sleep(0.1)
                print("Receiving SessionBegins ...")

                session_begins = await _ws.recv()
                print(session_begins)
                print("Sending messages ...")

                async def send():
                    try:
                        while st.session_state["run"]:
                            try:
                                data = stream.read(FRAMES_PER_BUFFER)
                                data = base64.b64encode(data).decode("utf-8")

                                json_data = json.dumps({"audio_data": str(data)})
                                await _ws.send(json_data)
                                await asyncio.sleep(0.01)

                            except websockets.exceptions.ConnectionClosedError as e:
                                print(f"Send connection closed: {e}")
                                break
                            except Exception as e:
                                print(f"Send error: {e}")
                                await asyncio.sleep(1)
                    except Exception as e:
                        print(f"Send function error: {e}")

                async def receive():
                    try:
                        while st.session_state["run"]:
                            try:
                                result_str = await _ws.recv()
                                result_json = json.loads(result_str)
                                result = result_json.get("text", "")

                                if result_json.get("message_type") == "FinalTranscript":
                                    print(result)

                                    st.session_state["text"] = f"<span style='color: orange;'>User:</span> {result}"
                                    st.markdown(st.session_state["text"], unsafe_allow_html=True)
                                    transcript.append(f"User: {result}")
                                    conversation_history.append({"role": "user", "content": result})

                                    if result:
                                        messages = [
                                            {"role": "system", "content": "You are a helpful assistant."}
                                        ] + list(conversation_history)

                                        try:
                                            chat_completion = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=messages,
                                                temperature=0.5,
                                                max_tokens=300,
                                            )
                                            reply = chat_completion.choices[0].message.content
                                            print(f"InterviewGPT: {reply}")
                                            play_tts(reply)
                                            conversation_history.append({"role": "assistant", "content": reply})
                                            transcript.append(f"InterviewGPT: {reply}")

                                            st.session_state["chatText"] = f"<span style='color: green;'>InterviewGPT:</span> {reply}"
                                            st.markdown(st.session_state["chatText"], unsafe_allow_html=True)
                                        except Exception as e:
                                            print(f"Error in chat completion: {e}")
                                            st.error("Error getting response from InterviewGPT")

                            except websockets.exceptions.ConnectionClosedError as e:
                                print(f"Receive connection closed: {e}")
                                break
                            except Exception as e:
                                print(f"Receive error: {e}")
                                await asyncio.sleep(1)
                    except Exception as e:
                        print(f"Receive function error: {e}")

                await asyncio.gather(send(), receive())
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            st.error("Error connecting to speech recognition service")

    if st.session_state["run"]:
        try:
            asyncio.run(send_receive())
        except Exception as e:
            print(f"Main loop error: {e}")
            st.error("An error occurred in the main application loop")

