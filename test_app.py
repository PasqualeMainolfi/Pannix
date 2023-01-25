import pannix as px
import numpy as np
import customtkinter as ctk
import tkdial as tkd
import pyaudio as pa
import wave
import threading
import tools


FILE = "./audio_file/vox.wav"
CHNLS = 2
CHUNK = 512

def play():
    global play_audio, degree, pan

    audio_file = wave.open(FILE, "rb")
    frame = audio_file.readframes(CHUNK)
    
    audio_engine = pa.PyAudio()
    stream = audio_engine.open(
        format=pa.get_format_from_width(audio_file.getsampwidth()),
        channels=CHNLS,
        rate=audio_file.getframerate(),
        output=True,
        frames_per_buffer=CHUNK
    )

    stream.start_stream()

    while play_audio:
        out = tools.to_nchnls(input_sig=frame, nchnls=CHNLS, format=np.int16)
        source = pan.pol_to_cart(rho=0.7, phi=degree, mode="deg")
        gains = pan.calculate_gains(source=source, normalize=True, mode="ray")
        for i in range(CHNLS):
             out[:, i] = out[:, i] * gains[i]
        out_frame = out.tobytes()
        stream.write(out_frame)
        if frame == b'':
            audio_file.rewind()
        frame = audio_file.readframes(CHUNK)
    
    stream.stop_stream()
    stream.close()
    audio_engine.terminate()

def play_button_function():
    global play_audio, thread
    play_audio = True
    thread = threading.Thread(target=play)
    thread.start()

def stop_button_function():
    global play_audio, thread
    play_audio = False
    thread.join()

def pan_dial_function():
    global degree
    degree = pan_dial.get()

play_audio = False
thread = None
degree = 0
pan = px.VBAP(loudspeaker_loc=[90, 0])

ctk.set_appearance_mode("dark")
win = ctk.CTk()
win.geometry("600x340")
win.title("Pannix")

pan_dial = tkd.ScrollKnob(
    master=win, 
    bg="#212325", 
    text="°", 
    steps=1, 
    radius=200,
    start=0,
    end=360, 
    bar_color="#212325", 
    progress_color="white", 
    outer_color="yellow", 
    outer_length=10, 
    border_width=30, 
    start_angle=0,
    inner_width=0, 
    outer_width=5, 
    text_font="calibri 20", 
    text_color="white", 
    fg="#212325",
    command=pan_dial_function
)
pan_dial.set(0)

play_button = ctk.CTkButton(
    master=win,
    text="Play",
    command=play_button_function
    )

stop_button = ctk.CTkButton(
    master=win,
    text="Stop",
    command=stop_button_function
    )

play_button.place(x=150, y=50)
stop_button.place(x=300, y=50)
pan_dial.place(x=197, y=100)
win.mainloop()

