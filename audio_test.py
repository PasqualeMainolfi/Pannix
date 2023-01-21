import pannix as px
import numpy as np
import pyaudio as pa
import wave
import tools


CHUNK = 512
CHNLS = 2
SAMPLE_RATE = 22050
FILE = "./audio_file/vox.wav"

p = pa.PyAudio()

pan = px.VBAP(loudspeaker_loc=[45, -45])
source = pan.pol_to_cart(rho=0.7, phi=-45, mode="deg")
g = pan.calculate_gains(source=source, normalize=True, mode="ray")
# pan.display_panning(source=source)

print(g)

audio_file = wave.open(FILE, "rb")
file_data = audio_file.readframes(CHUNK)

# while file_data:
#     print(np.frombuffer(file_data, dtype=np.uint16))
#     file_data = audio_file.readframes(CHUNK)
# print(np.frombuffer(file_data, dtype=np.float32).shape)

# init audio
stream = p.open(
    format=pa.paInt16,
    channels=CHNLS,
    output=True,
    frames_per_buffer=CHUNK,
    rate=SAMPLE_RATE
)

stream.start_stream()
frames = []
while file_data:
    out = tools.to_nchnls(input_sig=file_data, nchnls=CHNLS, format=np.int16)
    out = np.nan_to_num(out)
    for i in range(CHNLS):
        out[:, i] = out[:, i] * g[i]
    y = out.tobytes()
    stream.write(y)
    frames.append(y)
    file_data = audio_file.readframes(CHUNK)

stream.stop_stream()
stream.close()
p.terminate()
audio_file.close()

w = wave.open("testmultichn.wav", "wb")
w.setnchannels(CHNLS)
w.setsampwidth(p.get_sample_size(pa.paInt16))
w.setframerate(SAMPLE_RATE)
w.writeframes(b''.join(frames))
w.close()




