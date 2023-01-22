import pannix as px
import numpy as np
import pyaudio as pa
import wave
import tools


FILE = "./audio_file/vox.wav"
CHUNK = 512

audio_file = wave.open(FILE, "rb")
file_data = audio_file.readframes(CHUNK)

SAMPLE_RATE = audio_file.getframerate()
CHNLS = 2

p = pa.PyAudio()

# init audio
stream = p.open(
    format=pa.paInt16,
    channels=CHNLS,
    output=True,
    frames_per_buffer=CHUNK,
    rate=SAMPLE_RATE
)

stream.start_stream()

pan = px.VBAP(loudspeaker_loc=[90, 0])
# source = pan.pol_to_cart(rho=0.7, phi=-45, mode="deg")
# g = pan.calculate_gains(source=source, normalize=True, mode="ray")
# pan.display_panning(source=source)

angle = 0
step = 0.1
while file_data:
    out = tools.to_nchnls(input_sig=file_data, nchnls=CHNLS, format=np.int16)
    out = np.nan_to_num(out)
    source = pan.pol_to_cart(rho=np.random.rand(), phi=angle, mode="deg")
    g = pan.calculate_gains(source=source, normalize=True, mode="ray")
    angle += step
    angle %= 90
    for i in range(CHNLS):
        out[:, i] = out[:, i] * g[i]
    y = out.tobytes()
    stream.write(y)
    file_data = audio_file.readframes(CHUNK)

stream.stop_stream()
stream.close()
p.terminate()
audio_file.close()




