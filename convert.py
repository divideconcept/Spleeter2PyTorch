input_folders={2:'2stems', 4:'4stems', 5:'5stems'} #you can optinally only keep the entries you need
input_shape=(1,2,512,1536) #used for tracing (B x C x T x F)
output_folder='' #where to store the traced models and the unmixed wav files (if an example file is provided)
unmix_example='unmix.wav' #must be a wav file, can be empty if no example is provided

import torch
from torch import nn
import torch.nn.functional as F

from unet import UNet
from util import tf2pytorch

import math
import torchaudio

def load_ckpt(model, ckpt):
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            target_shape = state_dict[k].shape
            assert target_shape == v.shape
            state_dict.update({k: torch.from_numpy(v)})
        else:
            print('  ignore', k)

    model.load_state_dict(state_dict)
    return model

def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T

    Args:
        tensor(Tensor): BxCxFxL

    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    old_size = tensor.size(3)
    new_size = math.ceil(old_size/T) * T
    tensor = F.pad(tensor, [0, new_size - old_size])
    [b, c, t, f] = tensor.shape
    split = new_size // T
    return torch.cat(torch.split(tensor, T, dim=3), dim=0)


class SpleeterNet(nn.Module):
    def __init__(self, num_instruments, checkpoint_path):
        super().__init__()

        self.num_instruments=num_instruments

        ckpts = tf2pytorch(checkpoint_path, num_instruments)

        for i in range(self.num_instruments):
            sub_model = UNet(elu=False if num_instruments==2 else True, keras=True)
            sub_model.eval()
            load_ckpt(sub_model, ckpts[i])
            self.__setattr__('sub_model'+str(i), sub_model)

    def forward(self, x):
        result = []

        outputs = []
        for i in range(self.num_instruments):
            output=self.__getattr__('sub_model'+str(i))(x)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=-1)

        if self.num_instruments==5:
            mask = torch.softmax(outputs, dim=-1)
        else:
            mask = torch.sigmoid(outputs)

        for i in range(self.num_instruments):
            result.append(mask[..., i])

        return result


#OPTIONAL
class Estimator(nn.Module):
    def __init__(self, model):
        super(Estimator, self).__init__()

        self.model=model
        self.model.eval()

        # stft config
        self.F = input_shape[3]
        self.T = input_shape[2]
        self.win_length = 4096
        self.hop_length = 1024
        self.win = torch.hann_window(self.win_length)

    def compute_stft(self, wav):
        """
        Computes stft feature from wav

        Args:
            wav (Tensor): B x L
        """

        stft = torch.stft(
            wav, self.win_length, hop_length=self.hop_length, window=self.win)

        # only keep freqs smaller than self.F
        stft = stft[:, :self.F, :, :]
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        mag = torch.sqrt(real ** 2 + im ** 2)

        return stft, mag

    def inverse_stft(self, stft):
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        wav = torch.istft(stft, self.win_length, hop_length=self.hop_length,
                    window=self.win)
        return wav.detach()

    def separate(self, wav, separation_exponent=1):
        """
        Separates stereo wav into different tracks corresponding to different instruments

        Args:
            wav (tensor): 2 x L
        """

        # stft - 2 X F x L x 2
        # stft_mag - 2 X F x L
        stft, stft_mag = self.compute_stft(wav)

        L = stft.size(2)

        # 1 x 2 x F x T
        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
        stft_mag = pad_and_partition(stft_mag, self.T)  # B x 2 x F x T
        stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F

        B = stft_mag.shape[0]

        # compute instruments' mask
        masks = self.model(stft_mag)

        # compute denominator
        mask_sum = sum([m ** 2 for m in masks])
        mask_sum += 1e-10

        wavs = []
        for mask in masks:
            mask = (mask ** separation_exponent + 1e-10/2)/(mask_sum)
            mask = mask.transpose(2, 3)  # B x 2 X F x T

            mask = torch.cat(
                torch.split(mask, 1, dim=0), dim=3)

            mask = mask.squeeze(0)[:,:,:L].unsqueeze(-1) # 2 x F x L x 1
            stft_masked = stft * mask
            wav_masked = self.inverse_stft(stft_masked)

            wavs.append(wav_masked)

        return wavs


for input_folder in input_folders:
    print("")
    print("loading "+str(input_folder)+" stems model...")
    model=SpleeterNet(num_instruments=input_folder, checkpoint_path=input_folders[input_folder])
    model.eval()

    print("tracing "+str(input_folder)+" stems model...")
    traced_model=torch.jit.trace(model, torch.rand(input_shape), strict=False)
    print("saving "+str(input_folder)+" stems model...")
    torch.jit.save(traced_model, str(input_folder)+'stems.pt')

    if unmix_example:
        print("separating...")
        es = Estimator(traced_model) # test traced model
        wav, sr = torchaudio.load(unmix_example) # load wav audio
        wav_torch = wav / (wav.max() + 1e-8) # normalize audio
        wavs = es.separate(wav_torch)
        for i in range(len(wavs)):
            fname = str(input_folder)+'stems_out_{}.wav'.format(i)
            print('  writing '+fname+'...')
            torchaudio.save(output_folder+fname, wavs[i].squeeze(), sr, encoding="PCM_S", bits_per_sample=16)

print("done !")
