'''
DWT-SSIM implemetation based on pytorch-wavelets

References:
[1] https://github.com/fbcotter/pytorch_wavelets
[2] Mehul P. Sampat, et al, Complex Wavelet Structural Similarity: A New Image Similarity Index, IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 18, NO. 11, NOVEMBER 2009
[3] https://github.com/Po-Hsun-Su/pytorch-ssim
'''

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# import pdb
def _get_local_dwt_ssim(c1, c2, K=0.01):
    '''
    gets the local CW-SSIM
    :param c1: torch.tensor of shape (N, C, H1, W1), coefficents from windowed batch2
    :param c2: torch.tensor of shape (N, C, H1, W1), coefficents from windowed batch2
    :param K: float, constant to avoid divided by zero
    :return dwt_ssim: torch.tensor with 1 item
    '''
    assert c1.shape == c2.shape

    c1c2 = c1 * c2
    c1c2_sum_abs = torch.abs(torch.sum(c1c2))
    numerator = 2 * c1c2_sum_abs + K

    c1_squared = c1 ** 2
    c1_squared_sum = torch.sum(c1_squared)
    c2_squared = c2 ** 2
    c2_squared_sum = torch.sum(c2_squared)
    denominator = c1_squared_sum + c2_squared_sum + K

    dwt_ssim = numerator / denominator
    
    return dwt_ssim

class DWT_SSIM(nn.Module):
    '''
    Computes DWT-SSIM index for two given batches
    '''
    def __init__(self, J=3, wave='haar', window_size=15, stride=8, verbose=False):
        '''
        :param J: int, decomposition levels
        :param wave: wavelet to use,
                     see https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
                     for a list of available wavelets
        :param window_size: int, the window which the local CW-SSIM index is calculated
        :param stride: int, controls how much will the window move in one single step

        In general, local index masked by window is calculated first. The window
        the strides through the whole image scale to get a list of local indices.

        Formula (9) in reference [2] is used to compute the local index.

        The index returned by self.forward is the mean of those local indices.
        '''
        super().__init__()
        self.window_size = window_size
        self.J = J
        self.stride = stride
        self.dwt = DWTForward(J=J, wave=wave)
        self.verbose = verbose
    
    def forward(self, batch1, batch2):
        '''
        :param batch1: torch.tensor, source batch of shape (N, C, H, W)
        :param batch2: torch.tensor, target batch of shape (N, C, H, W)
        :return: CW-SSIM index for current batch
        '''

        height = batch1.shape[2]
        width = batch1.shape[3]

        window_count = 0
        dwt_ssim_Yl = 0.0
        dwt_ssim_Yh = 0.0
        dwt_ssim = 0.0
        for up in range(0, height, self.stride):
            possible_down = up + self.window_size
            down =  possible_down if possible_down <= height else height
            for left in range(0, width, self.stride):
                window_count += 1

                possible_right = left + self.window_size
                right = possible_right if possible_right <= width else width

                batch1_roi = batch1[:, :, up:down, left:right]
                batch2_roi = batch2[:, :, up:down, left:right]

                Yl1, Yh1 = self.dwt(batch1_roi)
                Yl2, Yh2 = self.dwt(batch2_roi)
                # pdb.set_trace()
                
                # approximation
                dwt_ssim_this_Yl = _get_local_dwt_ssim(Yl1, Yl2)
                if self.verbose: print(f"this_Yl = {dwt_ssim_this_Yl}")

                # detailed
                dwt_ssim_these_subbands = 0.0
                for subband1_Yh, subband2_Yh in zip(Yh1, Yh2):
                    this_band = _get_local_dwt_ssim(subband1_Yh, subband2_Yh)
                    dwt_ssim_these_subbands += this_band
                    if self.verbose: print(f"this_band = {this_band}")
                dwt_ssim_these_subbands /= self.J

                dwt_ssim += 0.2*dwt_ssim_this_Yl + 0.8*dwt_ssim_these_subbands # in favour of high frequency similarity

        dwt_ssim /= window_count

        return dwt_ssim