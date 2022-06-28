import numpy as np
import librosa
import soundfile as sf
import pathlib
import tensorflow as tf

from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt


DATAPATH = "data"
output_path = "output"
data_dir = pathlib.Path(DATAPATH)
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
print("NUMBER OF FILE : ", len(filenames))


def single_image():
    file_path = "/home/cavid/Desktop/Alienide/All_project/Alienide_Project/Hamba_project/Audio_classification/data/cow_original_data/1-69641-A.wav"
    file_name = file_path.split("/")[-1]
    audio_format = "wav"
    print(audio_format)
    sound = AudioSegment.from_file(file_path, format=audio_format)
    print(sound.dBFS)
    plt.plot(sound.get_array_of_samples())
    # plt.show()

    audio_chunks = split_on_silence(
        sound, min_silence_len=10, silence_thresh=-30, keep_silence=0)
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    plt.plot(combined.get_array_of_samples())
    plt.show()
    return combined


def save_file():
    for y in filenames:
        print(y)
        file_name = y.split("/")[-1]
        print(file_name)
        audio_format = "wav"
        print(audio_format)
        sound = AudioSegment.from_file(y, format=audio_format)
        print(sound.dBFS)
        plt.plot(sound.get_array_of_samples())
        # plt.show()

        audio_chunks = split_on_silence(
            sound, min_silence_len=10, silence_thresh=-30, keep_silence=0)
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
        plt.plot(combined.get_array_of_samples())
        plt.show()
        combined.export(output_path + "/" +
                        file_name, format=audio_format)


if __name__ == '__main__':
    save_file()
    # single_image()
