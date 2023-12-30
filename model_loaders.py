import file_check
import torch
import cv2

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from models import Wav2Lip
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

class model_loaders:

    def __init__(self, restorer, weight):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weight = weight
        # self.wav2lip_model = self.load_wav2lip_model()
        if restorer == 'GFPGAN':
            self.restorer = self.load_gfpgan_model()
        elif restorer == 'CodeFormer':
            self.restorer = self.load_codeformer_model()

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_wav2lip_model(self):
        model = Wav2Lip()
        print(f"Load checkpoint from: {file_check.WAV2LIP_MODEL_PATH}")
        checkpoint = self._load(file_check.WAV2LIP_MODEL_PATH)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def load_wav2lip_gan_model(self):
        model = Wav2Lip()
        print(f"Load checkpoint from: {file_check.WAV2LIP_GAN_MODEL_PATH}")
        checkpoint = self._load(file_check.WAV2LIP_GAN_MODEL_PATH)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def load_gfpgan_model(self):

        gfpgan = GFPGANv1Clean(
                        out_size=512,
                        num_style_feat=512,
                        channel_multiplier=2,
                        decoder_load_path=None,
                        fix_decoder=False,
                        num_mlp=8,
                        input_is_latent=True,
                        different_w=True,
                        narrow=1,
                        sft_half=True)

        loadnet = torch.load(file_check.GFPGAN_MODEL_PATH)

        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        gfpgan.load_state_dict(loadnet[keyname], strict=True)
        restorer = gfpgan.eval()
        return restorer.to(self.device)

    def load_codeformer_model(self):
        print(f"Load checkpoint from: {file_check.CODEFORMERS_MODEL_PATH}")
        
        model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(self.device)

        ckpt_path = file_check.CODEFORMERS_MODEL_PATH
        checkpoint = torch.load(ckpt_path)['params_ema']
        model.load_state_dict(checkpoint)
        return model.eval()
    
    def restore_wGFPGAN(self, dubbed_face):
        dubbed_face = cv2.resize(dubbed_face.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_CUBIC)
        dubbed_face_t = img2tensor(dubbed_face, bgr2rgb=True, float32=True)
        normalize(dubbed_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        dubbed_face_t = dubbed_face_t.unsqueeze(0).to(self.device)
        
        try:
            output = self.restorer(dubbed_face_t, return_rgb=False, weight=self.weight)[0]
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = dubbed_face
        
        restored_face = restored_face.astype(np.uint8)
        return restored_face
    
    def restore_wCodeFormer(self, dubbed_face):
        dubbed_face = cv2.resize(dubbed_face.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_CUBIC)
        dubbed_face_t = img2tensor(dubbed_face, bgr2rgb=True, float32=True)
        normalize(dubbed_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        dubbed_face_t = dubbed_face_t.unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.restorer(dubbed_face_t, w=self.weight, adain=True)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = tensor2img(dubbed_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype(np.uint8)
        return restored_face