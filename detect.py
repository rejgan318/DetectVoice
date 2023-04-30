import torch
"""
https://github.com/snakers4/silero-vad

pip install torchaudio
pip install soundfile # for Windows
# pip install sox     # for Linux
pip install numpy
"""
SAMPLING_RATE = 16000
from pprint import pprint

torch.set_num_threads(1)
# download example
# torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

USE_ONNX = False # change this to True if you want to test onnx model
if USE_ONNX:
    pass
    # !pip install -q onnxruntime
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              # force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
# speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
# pprint(speech_timestamps)
#
# print("------------------------------------------------------------------")

# speech_probs = []
# window_size_samples = 512 # use 256 for 8000 Hz model
# for i in range(0, len(wav), window_size_samples):
#     chunk = wav[i: i+window_size_samples]
#     if len(chunk) < window_size_samples:
#         break
#     speech_prob = model(chunk, SAMPLING_RATE)
#     speech_probs.append(speech_prob.item())
#
# model.reset_states() # reset model states after each audio
# print(speech_probs[:10]) # first 10 chunks predicts
#
# print("------------------------------------------------------------------")

## using VADIterator class

vad_iterator = VADIterator(model)
wav = read_audio(f'en_example.wav', sampling_rate=SAMPLING_RATE)

window_size_samples = 512 # number of samples in a single audio chunk
intervals = []
t_start = None
for i in range(0, len(wav), window_size_samples):
    speech_dict = vad_iterator(wav[i: i+ window_size_samples], return_seconds=True)
    if speech_dict:
        if speech_dict.get('start'):
            # print(speech_dict['start'], end=' ')
            if t_start:
                print(f"🟦 {speech_dict['start'] - intervals[-1][1]:3.1f}")
            t_start = speech_dict['start']
        else:
            print(f"{t_start:5.1f} ◀️▶️ {speech_dict['end']:5.1f} ({speech_dict['end'] - t_start:.1f}) ",  end=' ')
            intervals.append((t_start, speech_dict['end']))

voice_len = 0
for i in intervals:
    voice_len += i[1] - i[0]
total_len = len(wav) / SAMPLING_RATE
print(f"\nVoice length ∑ {voice_len:.1f}s ({voice_len/total_len*100:.1f}%) \nTotal length ∑ {total_len:.1f}s")

vad_iterator.reset_states() # reset model states after each audio
print("\nDone.")