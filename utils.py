import librosa
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

# 绘制语谱图和波形图
def draw_spectrogram_and_waveform(wav_data, fs):
    plt.subplot(2, 2, 1)
    plt.title("语谱图", fontsize=15)
    plt.specgram(wav_data, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
    plt.xlabel('秒/s', fontsize=15)
    plt.ylabel('频率/Hz', fontsize=15)

    plt.subplot(2, 2, 2)
    plt.title("波形图", fontsize=15)
    time = np.arange(0, len(wav_data)) * (1.0 / fs)
    plt.plot(time, wav_data)
    plt.xlabel('秒/s', fontsize=15)
    plt.ylabel('振幅', fontsize=15)

    plt.tight_layout()
    plt.show()

# 时域增强
# 噪声增强 - 控制噪声因子
def add_noise1(x, w=0.004):
    # w：噪声因子
    output = x + w * np.random.normal(loc=0, scale=1, size=len(x))
    return output

# 噪声增强 - 控制信噪比
def add_noise2(x, snr):
    """
    :param x:纯净语音
    :param snr: 信噪比
    :return: 生成执行信噪比的带噪语音
    """
    P_signal = np.mean(x**2)    # 信号功率
    k = np.sqrt(P_signal / 10 ** (snr / 10.0))  # 噪声系数 k
    return x + np.random.randn(len(x)) * k

# 噪声增强 - 纯净语音+噪声(指定信噪比)
def snr2noise(clean, noise, SNR):
    """
    :param clean: 纯净语音
    :param far_echo: 噪音
    :param SER: 指定的SNR
    :return: 根据指定的SNR求带噪语音(纯净语音+噪声)
    """
    p_clean = np.mean(clean ** 2)  # 纯净语音功率
    p_noise = np.mean(noise ** 2)  # 噪声功率

    scalar = np.sqrt(p_clean / (10 ** (SNR / 10)) / (p_noise + np.finfo(np.float32).eps))
    noisy = clean + scalar * noise

    return noisy

