import numpy as np
import matplotlib.pyplot as plt


iq = np.fromfile("output_signal.complex", dtype=np.complex64)


n = len(iq)
divider = 3


real = np.random.randn(n).astype(np.float32)
imag = np.random.randn(n).astype(np.float32)

real = real  / divider
imag = imag / divider

z = (real + 1j * imag).astype(np.complex64)






plt.plot(np.real(z[:5000]))
plt.plot(np.real(iq[:5000]))

plt.title("real part")
plt.show()


print('z[0:8]=', z[:8])