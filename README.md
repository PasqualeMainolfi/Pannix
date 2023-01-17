### **2D PANNER**  

Python implementation of 2D VBAP and DBAP algorithms, from:   
- V. Pulkki, *Virtual Sound Source Positioning Using Vector Base Amplitude Panning*
- T. Lossius, P. Baltazar, *DBAP Distance-Based Amplitude Panning*
- J. Sundstrom, *Speaker Placement Agnosticism: Improving the Distance-Based Amplitude Panning Algorithm*

moreover, the active arc searching function (2D) has been modified, implementing a line-line intersection algortithm
which allows for speeding up the computation (see ray_intersection.py).

reference: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection  

If you use this version of VBAP algo, please cite me.

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
# with mode argument specify which version you want to use: 
# mode="default" -> pulkki version
# mode="ray" -> version with line-line intersection algo 
v_gains = v.calculate_gains(source=[xs, ys], normalize=False, mode="ray")

print(d_gains)
print(v_gains)
```

the output is
```console
DBAP g = [0.09354905 0.98908099 0.09354905 0.06493032]
VBAP g = [0.18629955 0.54066234 0. 0. 0. 0. 0. 0. 0. 0.]

```
plot loudspeaker layout and relativa amp

```python
v.display_panning(source=(xs, ys))
```

testing performance:  
```python
for i in range(100000):
    v_gains = v.calculate_gains(source=[xs, ys], normalize=False, mode="default")
```
```console
computation time: 22.52220106124878 sec.
```

```python
for i in range(100000):
    v_gains = v.calculate_gains(source=[xs, ys], normalize=False, mode="ray")
```
```console
computation time: 1.339257001876831 sec.
```
so...

**1581.693732396861% increase in performance  
16.81693732396861x faster**
