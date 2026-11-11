"""
Generates synthetic speech-like WAV files for test cases in samples/
"""

import os
import numpy as np
import scipy.io.wavfile as wav

os.makedirs(r'D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\samples', exist_ok=True)
sample_rate = 16000

def create_speech_audio(duration=3.5, base_freq=180, noise_level=0.01):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Formants for speech-like harmonic spectrum
    f1 = np.sin(2 * np.pi * base_freq * t)
    f2 = 0.5 * np.sin(2 * np.pi * (base_freq * 2.2) * t)
    f3 = 0.25 * np.sin(2 * np.pi * (base_freq * 3.8) * t)
    # Syllabic envelope modulation (approx 4 syllables per second)
    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))
    audio = (f1 + f2 + f3) * env
    # Add light background noise
    noise = np.random.normal(0, noise_level, len(t))
    signal = audio + noise
    # Normalize to 16-bit PCM
    signal = signal / np.max(np.abs(signal)) * 0.9
    return (signal * 32767).astype(np.int16)

samples = {
    'sample1_meeting.wav': (3.2, 190, 0.008),
    'sample2_deploy.wav': (4.5, 175, 0.012),
    'sample3_medical.wav': (3.8, 160, 0.020),
    'sample4_smarthome.wav': (2.8, 210, 0.005)
}

for name, (dur, freq, noise) in samples.items():
    pcm = create_speech_audio(dur, freq, noise)
    out_path = os.path.join(r'D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\samples', name)
    wav.write(out_path, sample_rate, pcm)
    print(f'Generated {name}: duration={dur}s, path={out_path}')

print('All samples generated successfully!')
