### **2D PANNER**  

Python implementation of VBAP and DBAP algorithms, from:   
- V. Pulkki, *Virtual Sound Source Positioning Using Vector Base Amplitude Panning*
- T. Lossius, P. Baltazar, *DBAP Distance-Based Amplitude Panning*
- J. Sundstrom, *Speaker Placement Agnosticism: Improving the Distance-Based Amplitude Panning Algorithm*

Example:  
```python
import panner as p
import numpy as np

# when you initialize a panner, you can pass a list 
# with the exact position of the speakers or you can also just specify the number of speakers

# DBAP init -> in this case pass a list. If use a dbap panner you have to specify a rolloff coefficient and speaker weights
d = p.DBAP(loudspeaker_loc=LS, rolloff=6, weights=1)

# VBAP init -> in this case specify the number of speakers
v = p.VBAP(loudspeaker_num=10)

# source position
phi = np.pi/4
xs = 0.7 * np.cos(phi)
ys = 0.7 * np.sin(phi)

# reference point
xref, yref = xs, ys

# calculate gains
# DBAP -> if ref is None, reference point will be a geometric center. If spatial_blur is None will be default value
d_gains = d.calculate_gains(source=[xs, ys], ref=None, spatial_blur=0.1)

# VBAP
v_gains = v.calculate_gains(source=[xs, ys], normalize=False)

print(d_gains)
print(v_gains)
```

the output is
```console
DBAP g = [0.09354905 0.98908099 0.09354905 0.06493032]
VBAP g = [0.18629955 0.54066234 0. 0. 0. 0. 0. 0. 0. 0.]

```
plot loudspeaker layout (for example plot vbap speaker layout)

```python
d.plot_loudspeaker_loc(source=(xs, ys), g=d_gains)
```
