import numpy as np
import librosa
import librosa.filters
import soundfile
import math
import warnings
from scipy import signal

# Modified from the class TacotronHelper of the speech chain by andros
class TacotronAudio() :
    def __init__(self, config={"num_mels":80,
                               "sample_rate":16000,
                               "frame_length_ms":50,
                               "frame_shift_ms":12.5}) :
        """
        Feature processing of audios by Tacotron, including the methods 
        such as load_wav, save_wav, melspectrogram, spectrogram, inv_spectrogram...

        Parameters
        ---------
        config: a dictionary of feature configuration. You can pass a new dict to update the default configuration
        Default as following:
        {"pkg":"taco",
        "type":"mel",
        "num_mels":80,
        "num_freq":1025,
        "sample_rate":16000,
        "frame_length_ms":50,
        "frame_shift_ms":12.5,
        "preemphasis":0.97,
        "min_level_db":-100,
        "ref_level_db":20,
        "griffin_lim_iters":60,
        "power":1.5}
        """
        default_config = {"pkg":"taco",
                          "type":"mel",
                          "num_mels":80,
                          "num_freq":1025,
                          "sample_rate":16000,
                          "frame_length_ms":50,
                          "frame_shift_ms":12.5,
                          "preemphasis":0.97,
                          "min_level_db":-100,
                          "ref_level_db":20,
                          "griffin_lim_iters":60,
                          "power":1.5}
        default_config.update(config)
        self.config = default_config

    def load_wav(self, path, begin_sec=None, end_sec=None):
        """
        Load a wav file or its segment.

        Parameters
        ----------
        path: the path of the audio
        begin_sec: the beginning time (in second) of the segment of the audio
        end_sec: the end time (in second) of the segment of the audio
        
        Returns
        -------
        the data of wav file or the data of its segment
        """
        wav = librosa.core.load(path, sr=self.config['sample_rate'])[0]
        if begin_sec is None and end_sec is None: return wav
        else:
            if begin_sec is None: begin_index = 0
            if end_sec is None: end_index = len(wav)
            assert begin_sec < end_sec, "the begin time of the segment should be less than the end time."
            begin_index = max(int(begin_sec * self.config['sample_rate']), 0)
            end_index = min(int(end_sec * self.config['sample_rate']), len(wav))

            return wav[begin_index:end_index]

    def save_wav(self, wav, path) :
        """
        save a waveform data to an audio file at path.

        Parameters
        ----------
        path: the path of the audio
        """
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        try :
            librosa.output.write_wav(path, wav.astype(np.int16), self.config['sample_rate'])
        except :
            soundfile.write(path, wav.astype(np.int16), self.config['sample_rate'])

    def spectrogram(self, y):
        """
        Convert a waveform to its spectrogram
        
        Parameters
        ----------
        y: the waveform [num_samples]
        
        Returns
        -------
        the spectrogram [num_freq, num_frames] (Note: num_frames is computed by padded signal in librosa's stft, so approximated by num_samples / num_samples_of_a_frame_shift)
        """
        D = self._stft(self.preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.config['ref_level_db']
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Convert a spectrogram to its waveform using librosa, paired with spectrogram function.
        
        Parameters
        ----------
        spectrogram: the spectrogram [num_freq, num_frames]

        Returns
        ---------
        waveform [num_samples] (Note: the num_samples may not equals to original signal due to padding at stft transformation)
        """
        S = self._db_to_amp(self._denormalize(spectrogram) + self.config['ref_level_db']) # Convert back to linear
        return self.inv_preemphasis(self._griffin_lim(S ** self.config['power'])) # Reconstruct phase

    def melspectrogram(self, y):
        """
        Convert a wavform to its melspectrogram.
        
        Parameters
        ----------
        y: the waveform [num_samples]
        
        Returns
        -------
        the melspectrogram of the waveform [num_mels, num_frames] (Note: num_frames is computed by padded signal in librosa's stft, so approximated by num_samples / num_samples_of_a_frame_shift)
        """
        D = self._stft(self.preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.config['ref_level_db']
        return self._normalize(S)

    def preemphasis(self, x):
        return signal.lfilter([1, -self.config['preemphasis']], [1], x)

    def inv_preemphasis(self, x):
        return signal.lfilter([1], [1, -self.config['preemphasis']], x)

    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.config['griffin_lim_iters']):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)

    def _stft_parameters(self):
        n_fft = (self.config['num_freq'] - 1) * 2
        hop_length = int(self.config['frame_shift_ms'] / 1000 * self.config['sample_rate'])
        win_length = int(self.config['frame_length_ms'] / 1000 * self.config['sample_rate'])
        return n_fft, hop_length, win_length

    # Conversions:
    def _linear_to_mel(self, spectrogram):
        _mel_basis = None
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self):
        n_fft = (self.config['num_freq'] - 1) * 2
        return librosa.filters.mel(self.config['sample_rate'], n_fft, n_mels=self.config['num_mels'])

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.config['min_level_db']) / -self.config['min_level_db'], 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.config['min_level_db']) + self.config['min_level_db']
