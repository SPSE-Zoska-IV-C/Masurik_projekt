
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple, Optional

import numpy as np


def read_complex64_file(path: str) -> np.ndarray:
	data = np.fromfile(path, dtype=np.complex64)
	return data


def smooth_signal(x: np.ndarray, kernel_size: int) -> np.ndarray:
	if kernel_size <= 1:
		return x
	kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
	return np.convolve(x, kernel, mode="same")


def demodulate_envelope(envelope: np.ndarray, samples_per_bit: int, mode: str = "regular", num_bits: Optional[int] = None) -> Tuple[List[int], int, np.ndarray]:

	if envelope.size == 0:
		return [], 0, envelope

	
	env = envelope.astype(float)
	maxv = float(np.max(env))
	if maxv > 0:
		env = env / maxv

	kernel_size = max(1, int(samples_per_bit // 2))
	env_smooth = smooth_signal(env, kernel_size)

	L = len(env_smooth)
	best_score = -1.0
	best_offset = 0
	best_bits: List[int] = []

	
	
	for offset in range(0, max(1, samples_per_bit)):
		available = L - offset
		num_chunks = available // samples_per_bit
		if num_chunks <= 0:
			continue
		
		n_eval = num_chunks if num_bits is None else min(num_chunks, num_bits if mode == 'regular' else (num_bits // (2 if mode == 'mary' else 1)))

		means = []
		for i in range(n_eval):
			start = offset + i * samples_per_bit
			chunk = env_smooth[start:start + samples_per_bit]
			means.append(float(np.mean(chunk)) if chunk.size else 0.0)
		if len(means) == 0:
			continue
		means_arr = np.array(means)

		if mode == 'mary':
			
			bins = np.linspace(0.0, 1.0, 5)
			sym_idx = np.digitize(means_arr, bins) - 1
			sym_idx = np.clip(sym_idx, 0, 3)
			
			centers = np.array([0.0, 0.33, 0.66, 1.0])
			
			global_center = np.mean(centers)
			score = float(np.sum(np.abs(centers[sym_idx] - global_center)))
			if score > best_score:
				best_score = score
				best_offset = offset
				best_bits = []
				for s in sym_idx:
					if s == 0:
						best_bits.extend([0, 0])
					elif s == 1:
						best_bits.extend([0, 1])
					elif s == 2:
						best_bits.extend([1, 0])
					else:
						best_bits.extend([1, 1])
		else:
			
			local_min = float(np.min(means_arr))
			local_max = float(np.max(means_arr))
			local_thresh = (local_min + local_max) / 2.0
			score = float(np.sum(np.abs(means_arr - local_thresh)))
			if score > best_score:
				best_score = score
				best_offset = offset
				best_bits = [1 if m > local_thresh else 0 for m in means_arr]

	
	if len(best_bits) == 0:
		num_bits_rx = L // samples_per_bit
		all_means = []
		for i in range(num_bits_rx):
			chunk = env_smooth[i * samples_per_bit:(i + 1) * samples_per_bit]
			all_means.append(float(np.mean(chunk)) if chunk.size else 0.0)
		if mode == 'mary':
			if len(all_means) > 0:
				bins = np.linspace(0.0, 1.0, 5)
				sym_idx = np.digitize(np.array(all_means), bins) - 1
				sym_idx = np.clip(sym_idx, 0, 3)
				for s in sym_idx:
					if s == 0:
						best_bits.extend([0, 0])
					elif s == 1:
						best_bits.extend([0, 1])
					elif s == 2:
						best_bits.extend([1, 0])
					else:
						best_bits.extend([1, 1])
		else:
			if len(all_means) > 0:
				fb_thresh = (min(all_means) + max(all_means)) / 2.0
			else:
				fb_thresh = 0.5
			for m in all_means:
				best_bits.append(1 if m > fb_thresh else 0)

	return best_bits, best_offset, env_smooth


def bits_to_str(bits: List[int]) -> str:
	return ''.join(str(int(b)) for b in bits)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Demodulate complex64 ASK file")
	parser.add_argument("--infile", required=True, help="Path to complex64 input file")
	parser.add_argument("--samples-per-bit", type=int, default=128, help="Samples per symbol/bit")
	parser.add_argument("--mode", choices=["regular", "mary"], default="regular", help="regular (binary) or mary (4-ary)")
	parser.add_argument("--out", default=None, help="Write demodulated bits to this file (plain text)")
	parser.add_argument("--plot", action="store_true", help="Save an envelope+demod plot to demod_plot.png (requires matplotlib)")
	parser.add_argument("--max-bits", type=int, default=None, help="Maximum number of bits to demodulate (for limiting output)")

	args = parser.parse_args(argv)

	data = read_complex64_file(args.infile)
	if data.size == 0:
		print("Input file contains no samples", file=sys.stderr)
		return 2

	envelope = np.abs(data)

	demod_bits, offset, env_smooth = demodulate_envelope(envelope, args.samples_per_bit, args.mode, args.max_bits)

	
	if args.max_bits is not None:
		demod_bits = demod_bits[: args.max_bits]

	bitstr = bits_to_str(demod_bits)

	if args.out:
		with open(args.out, 'w') as f:
			f.write(bitstr)
		print(f"Wrote {len(demod_bits)} bits to {args.out}")
	else:
		print(bitstr)

	if args.plot:
		try:
			import matplotlib.pyplot as plt

			samples = np.arange(len(env_smooth))
			plt.figure(figsize=(10, 4))
			plt.plot(samples, env_smooth, label='Smoothed envelope')

			expanded = np.zeros_like(env_smooth, dtype=float)
			if args.mode == 'mary':
				n_symbols = len(demod_bits) // 2
				for i in range(n_symbols):
					b1 = demod_bits[2 * i]
					b2 = demod_bits[2 * i + 1]
					if b1 == 0 and b2 == 0:
						val = 0.0
					elif b1 == 0 and b2 == 1:
						val = 0.33
					elif b1 == 1 and b2 == 0:
						val = 0.66
					else:
						val = 1.0
					start = offset + i * args.samples_per_bit
					end = start + args.samples_per_bit
					if start >= len(expanded):
						break
					expanded[start:min(end, len(expanded))] = float(val)
			else:
				for i, b in enumerate(demod_bits):
					start = offset + i * args.samples_per_bit
					end = start + args.samples_per_bit
					if start >= len(expanded):
						break
					expanded[start:min(end, len(expanded))] = float(b)

			plt.plot(samples, expanded, label='Demodulated (expanded)')
			plt.title(f"Demodulation ({args.mode}) offset={offset} samples_per_bit={args.samples_per_bit}")
			plt.xlabel('Sample')
			plt.ylabel('Normalized amplitude')
			plt.legend()
			plt.tight_layout()
			outpng = 'demod_plot.png'
			plt.savefig(outpng)
			print(f"Saved plot to {outpng}")
		except Exception as e:
			print(f"Plotting failed: {e}", file=sys.stderr)

	return 0


if __name__ == '__main__':
	raise SystemExit(main())

