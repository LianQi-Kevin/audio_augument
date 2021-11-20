import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示符号


# 1. 读取音频文件，获取音频文件基本信息：采样个数，采样周期，与每个采样的声音信号值。绘制音频时域的：时间/位移图像
# 读取音频文件
sample_rate, noised_sigs = wf.read('wav_file/TTS_CAT_1.wav')
print(sample_rate)  # sample_rate：采样率44100
print(noised_sigs.shape)    # noised_sigs:存储音频中每个采样点的采样位移(220500,)
times = np.arange(noised_sigs.size) / sample_rate

plt.figure('Filter')
plt.subplot(221)
plt.title('Time Domain', fontsize=16)
plt.ylabel('Signal', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(times[:178], noised_sigs[:178], c='orangered', label='Noised')
plt.legend()

# 2. 基于傅里叶变换，获取音频频域信息，绘制音频频域的：频率/能量图像
# 傅里叶变换后，绘制频域图像
freqs = nf.fftfreq(times.size, times[1] - times[0])
complex_array = nf.fft(noised_sigs)
pows = np.abs(complex_array)

plt.subplot(222)
plt.title('Frequency Domain', fontsize=16)
plt.ylabel('Power', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
# 指数增长坐标画图
plt.semilogy(freqs[freqs > 0], pows[freqs > 0], c='limegreen', label='Noised')
plt.legend()

# 3. 将低频噪声去除后绘制音频频域的：频率/能量图像
# 寻找能量最大的频率值
fund_freq = freqs[pows.argmax()]
# where函数寻找那些需要抹掉的复数的索引
noised_indices = np.where(freqs != fund_freq)
# 复制一个复数数组的副本，避免污染原始数据
filter_complex_array = complex_array.copy()
filter_complex_array[noised_indices] = 0
filter_pows = np.abs(filter_complex_array)

plt.subplot(224)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(freqs[freqs >= 0], filter_pows[freqs >= 0], c='dodgerblue', label='Filter')
plt.legend()

# 4. 基于逆向傅里叶变换，生成新的音频信号，绘制音频时域的：时间/位移图像
filter_sigs = nf.ifft(filter_complex_array).real
plt.subplot(223)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Signal', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(times[:178], filter_sigs[:178], c='hotpink', label='Filter')
plt.legend()


# 5. 重新生成音频文件
wf.write('augumented_wav/filter.wav', sample_rate, filter_sigs)
plt.show()