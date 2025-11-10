from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from save_created_ask import CreateASK, gr, blocks
from save_created_ask_mary import CreateASKmary, gr, blocks
import numpy as np
import random

app = Dash(__name__)

app.layout = html.Div([
    html.H1("ASK Modulation Generator"),
    
    html.Div([
        html.Label("Modulation Type:"),
        dcc.Dropdown(
            id='mod-type',
            options=[
                {'label': 'Regular ASK (binary)', 'value': 'regular'},
                {'label': 'M-ary ASK (4-ary / M=4)', 'value': 'mary'}
            ],
            value='regular',
            clearable=False
        ),
        html.Label("Sampling Rate (Hz):"),
        dcc.Input(
            id='samp-rate',
            type='number',
            value=1280000,
            step=10000
        ),
        
        html.Label("Carrier Frequency (Hz):"),
        dcc.Input(
            id='frequency',
            type='number',
            value=500000,
            step=10000
        ),
        
        html.Label("Number of Bits:"),
        dcc.Input(
            id='num-bits',
            type='number',
            value=32,
            min=1,
            max=128,
            step=1
        ),
        
        html.Label("Noise Amplitude:"),
        dcc.Input(
            id='noise-amp',
            type='number',
            value=0.1,
            min=0,
            max=2,
            step=0.1
        ),
        
        html.Label("Samples per Bit:"),
        dcc.Input(
            id='samples-per-bit',
            type='number',
            value=128,
            min=1,
            max=1024,
            step=1
        ),
        
        html.Button('Generate Signal', id='generate-button', n_clicks=0),
        
        html.Div(id='output-bits'),
        
        html.Div([
            dcc.Graph(id='signal-plot')
            ,
            dcc.Graph(id='demod-plot')
        ])
    ])
])

@callback(
    [Output('output-bits', 'children'),
    Output('signal-plot', 'figure'),
    Output('demod-plot', 'figure')],
    Input('generate-button', 'n_clicks'),
    [State('samp-rate', 'value'),
     State('frequency', 'value'),
     State('num-bits', 'value'),
     State('noise-amp', 'value'),
     State('samples-per-bit', 'value')],
    State('mod-type', 'value'),
    prevent_initial_call=True
)
def generate_signal(n_clicks, samp_rate, frequency, num_bits, noise_amp, samples_per_bit, mod_type):
    if n_clicks > 0:
        
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        # If M-ary (4-ary) selected, ensure even number of bits (pairs)
        if mod_type == 'mary' and len(bits) % 2 == 1:
            bits.append(0)
        
        
        output_file = "output_signal.complex"
        
        
        # Choose generator class based on modulation type
        if mod_type == 'mary':
            tb = CreateASKmary(
                bits=bits,
                samp_rate=samp_rate,
                frequency=frequency,
                noise_amp=noise_amp,
                output_file=output_file,
                samples_per_bit=samples_per_bit
            )
        else:
            tb = CreateASK(
                bits=bits,
                samp_rate=samp_rate,
                frequency=frequency,
                noise_amp=noise_amp,
                output_file=output_file,
                samples_per_bit=samples_per_bit
            )
        
        tb.start()
        tb.wait()
        
        
        with open(output_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.complex64)
        
        
        samples = np.arange(len(data))
        
        
        fig = go.Figure()
        
        
        fig.add_trace(go.Scatter(
            x=samples,
            y=data.real,
            name='Real Part',
            mode='lines'
        ))
        
        
        fig.add_trace(go.Scatter(
            x=samples,
            y=data.imag,
            name='Imaginary Part',
            mode='lines'
        ))
        
        fig.update_layout(
            title=f'ASK Modulated Signal ({len(data)} Samples)',
            xaxis_title='Sample Number',
            yaxis_title='Amplitude',
            showlegend=True
        )

        
        envelope = np.abs(data)
        
        best_demod_bits = []
        if envelope.size == 0:
            demod_fig = go.Figure()
        else:
            
            envelope = envelope / np.max(envelope)

            
            
            kernel_size = max(1, int(samples_per_bit // 2))
            kernel = np.ones(kernel_size) / float(kernel_size)
            
            envelope_smooth = np.convolve(envelope, kernel, mode='same')

            
            
            
            L = len(envelope_smooth)
            best_score = -1.0
            best_offset = 0
            best_demod_bits = []
            
            
            
            for offset in range(0, max(1, samples_per_bit)):
                available = L - offset
                num_chunks = available // samples_per_bit
                if num_chunks <= 0:
                    continue
                
                n_eval = min(num_chunks, num_bits)
                means = []
                for i in range(n_eval):
                    start = offset + i * samples_per_bit
                    chunk = envelope_smooth[start:start + samples_per_bit]
                    means.append(float(np.mean(chunk)) if chunk.size else 0.0)
                if len(means) == 0:
                    continue
                means_arr = np.array(means)
                
                if mod_type == 'mary':
                    # Map means into 4 symbol indices (0..3) using equal-width bins
                    bins = np.linspace(0.0, 1.0, 5)  # edges for 4 bins
                    sym_idx = np.digitize(means_arr, bins) - 1
                    sym_idx = np.clip(sym_idx, 0, 3)
                    # Score by separation of symbol means from bin centers
                    centers = np.array([0.0, 0.33, 0.66, 1.0])
                    score = float(np.sum(np.abs(centers[sym_idx] - np.mean(centers))))
                    if score > best_score:
                        best_score = score
                        best_offset = offset
                        # Convert each symbol index to two bits
                        best_demod_bits = []
                        for s in sym_idx:
                            if s == 0:
                                best_demod_bits.extend([0, 0])
                            elif s == 1:
                                best_demod_bits.extend([0, 1])
                            elif s == 2:
                                best_demod_bits.extend([1, 0])
                            else:
                                best_demod_bits.extend([1, 1])
                else:
                    local_thresh = float((means_arr.min() + means_arr.max()) / 2.0)
                    score = float(np.sum(np.abs(means_arr - local_thresh)))
                    if score > best_score:
                        best_score = score
                        best_offset = offset
                        best_demod_bits = [1 if m > local_thresh else 0 for m in means_arr]

            
            if len(best_demod_bits) == 0:
                num_bits_rx = L // samples_per_bit
                best_demod_bits = []

                all_means = []
                for i in range(num_bits_rx):
                    chunk = envelope_smooth[i*samples_per_bit:(i+1)*samples_per_bit]
                    avg = float(np.mean(chunk)) if chunk.size else 0.0
                    all_means.append(avg)
                if mod_type == 'mary':
                    if len(all_means) > 0:
                        # quantize into 4 symbols
                        bins = np.linspace(0.0, 1.0, 5)
                        sym_idx = np.digitize(np.array(all_means), bins) - 1
                        sym_idx = np.clip(sym_idx, 0, 3)
                        for s in sym_idx:
                            if s == 0:
                                best_demod_bits.extend([0, 0])
                            elif s == 1:
                                best_demod_bits.extend([0, 1])
                            elif s == 2:
                                best_demod_bits.extend([1, 0])
                            else:
                                best_demod_bits.extend([1, 1])
                    else:
                        best_demod_bits = []
                else:
                    if len(all_means) > 0:
                        fb_thresh = float((min(all_means) + max(all_means)) / 2.0)
                    else:
                        fb_thresh = 0.5
                    for m in all_means:
                        best_demod_bits.append(1 if m > fb_thresh else 0)

            demod_fig = go.Figure()
            samples = np.arange(len(envelope_smooth))
            demod_fig.add_trace(go.Scatter(x=samples, y=envelope_smooth, name='Smoothed Envelope'))

            
            if len(best_demod_bits) > 0:
                expanded = np.zeros(len(envelope_smooth), dtype=float)
                if mod_type == 'mary':
                    # For plotting, reconstruct symbol index (0..3) per symbol period
                    n_symbols = len(best_demod_bits) // 2
                    for i in range(n_symbols):
                        b1 = best_demod_bits[2*i]
                        b2 = best_demod_bits[2*i + 1]
                        if b1 == 0 and b2 == 0:
                            val = 0.0
                        elif b1 == 0 and b2 == 1:
                            val = 0.33
                        elif b1 == 1 and b2 == 0:
                            val = 0.66
                        else:
                            val = 1.0
                        start = best_offset + i * samples_per_bit
                        end = start + samples_per_bit
                        if start >= len(expanded):
                            break
                        expanded[start:min(end, len(expanded))] = float(val)
                else:
                    for i, b in enumerate(best_demod_bits):
                        start = best_offset + i * samples_per_bit
                        end = start + samples_per_bit
                        if start >= len(expanded):
                            break
                        expanded[start:min(end, len(expanded))] = float(b)
                demod_fig.add_trace(go.Scatter(x=samples, y=expanded, name='Demodulated Bits', mode='lines'))

            demod_fig.update_layout(title='Demodulation', xaxis_title='Sample', yaxis_title='Amplitude')
            
            
            # For comparison, if mary mode we may have padded bits earlier; compare up to original requested length
            cmp_n = min(len(bits), len(best_demod_bits))
            if cmp_n > 0:
                matches = sum(1 for i in range(cmp_n) if bits[i] == best_demod_bits[i])
                match_pct = matches / cmp_n * 100.0
                match_text = f"{match_pct:.1f}% ({matches}/{cmp_n})"
            else:
                match_text = "N/A"
        
        return [
            html.Div([
                html.H3("Generated Bits:"),
                html.P(''.join(str(b) for b in bits)),
                html.H3("Demodulated Bits:"),
                html.P(''.join(str(b) for b in best_demod_bits)),
                html.H3("Match:"),
                html.P(match_text)
            ]),
            fig,
            demod_fig
        ]
    
    return [html.Div(), {}]

if __name__ == '__main__':
    app.run(debug=True)
