from utils import *
# from scipy.io import wavfile as scipywav
# import time
''' 
for Sky-Hackathon5
    ASR:
        请指出哪个图片的序列号是1/2/3/4
    TTS:
        猫/狗/马/人图片的序列号是1/2/3/4
'''

fs = 22050
input_wav = "wav_file/TTS_DOG_1.wav"
output_dir = "augumented_wav/"

# 加载语音文件
wav_data, sr = librosa.load(input_wav, sr=fs, mono=True)
# draw_spectrogram_and_waveform(wav_data, sr)
# librosa.output.write_wav(output_dir + 'source.wav', wav_data, fs)


# 噪声增强 - 控制噪声因子
Augmentation1 = add_noise1(x=wav_data, w=0.02)
draw_spectrogram_and_waveform(Augmentation1, sr)
librosa.output.write_wav(output_dir + 'add_noise1.wav', Augmentation1, fs)

# 噪声增强 - 控制信噪比
Augmentation2 = add_noise2(x=wav_data, snr=30)
# draw_spectrogram_and_waveform(Augmentation2, sr)
# librosa.output.write_wav(output_dir + 'add_noise2.wav', Augmentation2, fs)

# 噪声增强 - 纯净语音+噪声(指定信噪比)
noise = np.random.normal(loc=0, scale=1, size=len(wav_data))
Augmentation3 = snr2noise(clean=wav_data,noise=noise,SNR=0.2)
# draw_spectrogram_and_waveform(Augmentation3, sr)
# librosa.output.write_wav(output_dir + 'add_noise3.wav', Augmentation3, fs)

