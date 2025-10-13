#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# ASK Signal Generator (Headless)
# For dataset generation
#

from gnuradio import gr, blocks, analog, filter
from gnuradio.filter import firdes
import numpy as np
import random
import argparse
from gnuradio.fft import window


class CreateASK(gr.top_block):

    def __init__(self, bits, samp_rate=128000, frequency=500000,
                 noise_amp=1.5, output_file="output_signal.complex"):
        gr.top_block.__init__(self, "ASK Generator")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.frequency = frequency
        self.noise_amp = noise_amp
        self.bits = bits
        self.output_file = output_file

        ##################################################
        # Blocks
        ##################################################

        # Convert bits to complex (1+0j or 0+0j)
        symbols = np.array(bits, dtype=np.complex64)

        # Create finite (non-repeating) vector source
        self.src = blocks.vector_source_c(symbols.tolist(), False)

        # Repeat bits to create a longer waveform (1 bit = 128 samples)
        self.repeat = blocks.repeat(gr.sizeof_gr_complex, 128)

        # Low-pass filter (pulse shaping)
        self.lpf = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1.0, samp_rate,
                8000, 20000,
                window.WIN_HAMMING,
                6.76
            )
        )

        # Carrier signal
        self.carrier = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, frequency, 1, 0)

        # Multiply signal (ASK modulation)
        self.modulated = blocks.multiply_vcc(1)

        # Add Gaussian noise
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amp)
        self.adder = blocks.add_vcc(1)

        # File sink to save complex IQ data
        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, output_file, False)
        self.file_sink.set_unbuffered(True)
        
        ##################################################
        # Connections
        ##################################################
        self.connect(self.src, self.repeat)
        self.connect(self.repeat, self.lpf)
        self.connect(self.lpf, (self.modulated, 0))
        self.connect(self.carrier, (self.modulated, 1))
        self.connect(self.modulated, (self.adder, 0))
        self.connect(self.noise, (self.adder, 1))
        self.connect(self.adder, self.file_sink)


def main():
    parser = argparse.ArgumentParser(description="Generate ASK-modulated IQ samples")
    parser.add_argument("--outfile", type=str, default="output_signal.complex",
                        help="Output file for complex samples")
    parser.add_argument("--bits", type=str, default="random",
                        help="Bit sequence (e.g. 101010) or 'random'")
    parser.add_argument("--numbits", type=int, default=32,
                        help="Number of random bits if bits='random'")
    parser.add_argument("--samp-rate", type=float, default=128000,
                        help="Sample rate (Hz)")
    parser.add_argument("--freq", type=float, default=500000,
                        help="Carrier frequency (Hz)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise amplitude")

    args = parser.parse_args()

    # Generate bits
    if args.bits == "random":
        bits = [random.randint(0, 1) for _ in range(args.numbits)]
    else:
        bits = [int(b) for b in args.bits.strip()]

    print(f"Generating ASK signal -> {args.outfile}")
    print(f"Bits: {bits}")

    tb = CreateASK(bits, samp_rate=args.samp_rate,
                   frequency=args.freq,
                   noise_amp=args.noise,
                   output_file=args.outfile)

    tb.start()
    tb.wait()
    print("Done.")


if __name__ == "__main__":
    main()
