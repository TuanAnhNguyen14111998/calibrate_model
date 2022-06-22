import torch
import numpy as np

class FDA():
    def __init__(self, L=0.1):
        self.L = 0.1
    
    def extract_ampl_phase(self, fft_im):
        # fft_im: size should be bx3xhxwx2
        fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2(fft_im[:,:,:,:,1], fft_im[:,:,:,:,0])

        return fft_amp, fft_pha
    
    def low_freq_mutate(self, amp_src, amp_trg, L=0.1):
        _, _, h, w = amp_src.size()
        b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
        amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
        amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
        amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
        amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right

        return amp_src

    def fda_source_to_image(self, src_img, trg_img):
        """
        Input:
        - src_img: source image (Torch Tensor)
        - trg_img: target image (Torch Tensor)
        """
        # get fft of both source and target
        fft_src = torch.rfft(src_img.clone(), signal_ndim=2, onesided=False ) 
        fft_trg = torch.rfft(trg_img.clone(), signal_ndim=2, onesided=False )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = self.extract_ampl_phase(fft_src.clone())
        amp_trg, pha_trg = self.extract_ampl_phase(fft_trg.clone())

        # replace the low frequency amplitude part of source with that from target
        amp_src_ = self.low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=self.L)

        # recompose fft of source
        fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float )
        fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
        fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

        # get the recomposed image: source content, target style
        _, _, imgH, imgW = src_img.size()
        src_in_trg = torch.irfft(fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW])

        return src_in_trg
    
    def fda_source_to_target_np(self, src_img, trg_img, L=0.1 ):
        # exchange magnitude
        # input: src_img, trg_img

        src_img_np = src_img #.cpu().numpy()
        trg_img_np = trg_img #.cpu().numpy()

        # get fft of both source and target
        fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
        fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutate the amplitude part of source with target
        amp_src_ = self.low_freq_mutate_np(amp_src, amp_trg, L=self.L)

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src )

        # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg)

        return src_in_trg
        