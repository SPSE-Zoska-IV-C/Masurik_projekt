import numpy as np
import matplotlib.pyplot as plt

iq = np.fromfile("Testing_data3/000993.complex", dtype=np.complex64)

plt.plot(np.real(iq[:8000]))
plt.title("ASK signal (real part)")
plt.show()