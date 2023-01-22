import numpy as np
import wave
import pannix as px
import tools as t

FILE = "./audio_file/vox.wav"
CHNLS = 8
CHUNK = 512

# open file
af = wave.open(FILE, "rb")
sample_rate = af.getframerate()
ad = af.readframes(CHUNK)

# define VBAP panner
pan = px.VBAP(loudspeaker_num=CHNLS)

# apply pan
frames = []
frames_out = []
angle = 0
step = 0.5
while ad:
    y = t.to_nchnls(ad, nchnls=CHNLS, format=np.int16)

    source = pan.pol_to_cart(rho=0.7, phi=angle, mode="deg")
    g = pan.calculate_gains(source=source, mode="ray")

    for i in range(CHNLS):
        y[:, i] = y[:, i] * g[i]
    frames_out.append(y)
    frames.append(y.tobytes())
    
    angle += step
    angle %= 360

    ad = af.readframes(CHUNK)

t.save_audio_file(
    path="multich.wav", 
    frames=frames, 
    sample_rate=af.getframerate(),
    nchnls=CHNLS,
    sampwidth=2
)



# f = np.array(frames_out, dtype=object)

# out = f[0]
# for i in range(1, len(f)):
#     out = np.concatenate((out, f[i]))
# out = out.T

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(8, 1, figsize=(10, 7), constrained_layout=True)

# for i in range(CHNLS):
#     ax[i].plot(out[i])
# # plt.tight_layout()
# plt.show()


