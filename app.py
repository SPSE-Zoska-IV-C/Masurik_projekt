from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from save_created_ask import CreateASK, gr, blocks
import numpy as np
import random

app = Dash(__name__)

app.layout = html.Div([
    html.H1("ASK Modulation Generator"),
    
    html.Div([
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
    prevent_initial_call=True
)
def generate_signal(n_clicks, samp_rate, frequency, num_bits, noise_amp, samples_per_bit):
    if n_clicks > 0:
        
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        
        
        output_file = "output_signal.complex"
        
        
        class ModifiedCreateASK(CreateASK):
            def __init__(self, *args, samples_per_bit=128, **kwargs):
                super().__init__(*args, **kwargs)
                
                self.repeat = blocks.repeat(gr.sizeof_gr_complex, samples_per_bit)

        
        tb = ModifiedCreateASK(
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
                for i, b in enumerate(best_demod_bits):
                    start = best_offset + i * samples_per_bit
                    end = start + samples_per_bit
                    if start >= len(expanded):
                        break
                    expanded[start:min(end, len(expanded))] = float(b)
                demod_fig.add_trace(go.Scatter(x=samples, y=expanded, name='Demodulated Bits', mode='lines'))

            demod_fig.update_layout(title='Demodulation', xaxis_title='Sample', yaxis_title='Amplitude')
            
            
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
