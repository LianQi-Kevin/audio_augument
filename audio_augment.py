from utils import *

''' 
for Sky-Hackathon5
    ASR
        请指出哪个图片的序列号是1/2/3/4
    TTS
        猫/狗/马/人图片的序列号是1/2/3/4
'''

fs = 16000
input_wav = "wav_file/TTS_DOG_1.wav"
output_dir = "augumented_wav/"
# 加载语音文件
wav_data, _ = librosa.load(input_wav, sr=fs, mono=True)
draw_spectrogram_and_waveform(wav_data, fs)

# 噪声增强 - 控制噪声因子
Augmentation = add_noise1(x=wav_data, w=0.004)
draw_spectrogram_and_waveform(Augmentation, fs)
librosa.output.write_wav(output_dir + 'add_noise1.wav', wav_data, fs)

# 噪声增强 - 控制信噪比
Augmentation = add_noise2(x=wav_data, snr=50)
draw_spectrogram_and_waveform(Augmentation, fs)
librosa.output.write_wav(output_dir + 'add_noise2.wav', wav_data, fs)
