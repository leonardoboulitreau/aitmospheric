import torch
from torch.utils.data import DataLoader, Dataset
import time
import os
import numpy as np
import librosa
import librosa.display
from IPython.display import Audio
from natsort import natsorted
import matplotlib.pyplot as plt

class AugmentedESC(torch.utils.data.Dataset):
    def __init__(self, path_list, sample_rate=22050, lazy=False, n_fft=2048, win_len=2000, hop_len=500) -> None:
        super().__init__()
        self.paths = path_list
        self.sr = sample_rate
        self.lazy = lazy 
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len

        if not self.lazy:
            self.data = [self._load(i) for i in range(len(self))]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.lazy:
            return self._load(idx)
        else:
            return self.data[idx]

    def _load(self, idx):
        audio = librosa.load(self.paths[idx], sr=self.sr)[0]
        if self._preprocess is not None:
            audio = self._preprocess(audio)
        return audio

    def _preprocess(self, audio):
        if len(audio) == 0:
            audio = np.zeros(shape=[1], dtype=audio.dtype)
        return np.log1p(np.abs(librosa.stft(y = audio, n_fft=self.n_fft, win_length=self.win_len, hop_length=self.hop_len)))

    def _postprocess(self, spectrogram):
        return librosa.griffinlim(S=np.expm1(np.maximum(spectrogram, 0)), win_length=2000, hop_length=500, random_state=0)

    def play_audio(self, idx):
        audio = librosa.load(self.paths[idx], sr = self.sr)[0]
        return Audio(audio, rate=self.sr)

    def plot_spectrogram(self, idx):
        spectrogram = self[idx]
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(spectrogram,ref=np.max),
                                      y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        return None

def load_data_and_data_loaders(audio_folder_path, sample_rate, train_val_split, 
lazy, n_fft, win_len, hop_len, num_workers, batch_size):
    audio_paths = [audio_folder_path + name for name in natsorted(os.listdir(audio_folder_path))]
    train_audios_paths = audio_paths[0:int(train_val_split*len(audio_paths))]
    val_audios_paths = audio_paths[-int((1-train_val_split)*len(audio_paths))-1:]
    training_data = AugmentedESC(train_audios_paths, sample_rate, lazy, n_fft, win_len, hop_len)
    validation_data = AugmentedESC(val_audios_paths, sample_rate, lazy, n_fft, win_len, hop_len)
    training_loader = DataLoader(dataset=training_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    return training_data, validation_data, training_loader, validation_loader, #x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp, savepath, step):
    SAVE_MODEL_PATH = savepath

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + str(step) + '.pth')

class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset 
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading latent block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading latent block data')
        
        self.data = data[:-500] if train else data[-500:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)
           
def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val