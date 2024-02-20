# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

from helpers import file_check

import torch
import cv2
import os

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from models import Wav2Lip
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class ModelLoader:

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
        print(f"Loading wav2lip checkpoint from: {file_check.WAV2LIP_MODEL_PATH}")
        checkpoint = self._load(file_check.WAV2LIP_MODEL_PATH)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()
    
    def load_realesrgan_model(self, model_name, tile, half=False):

        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn("YAVU uses RealESRGAN for backgound upscaling and it's inference is really slow on CPU. Please consider using GPU.")
            bg_upsampler = None
        else:
            if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
            elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
            elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
            elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
            elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
            elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4

            realesrgan_model_path = os.path.join(file_check.REALESRGAN_WEIGHTS_DIR, f'{model_name}.pth')

            bg_upsampler = RealESRGANer(
                scale=netscale,
                model_path=realesrgan_model_path,
                dni_weight=None,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
                gpu_id=None)
            
        return bg_upsampler

    def load_wav2lip_gan_model(self):
        model = Wav2Lip()
        print(f"Loading wav2lip checkpoint from: {file_check.WAV2LIP_GAN_MODEL_PATH}")
        checkpoint = self._load(file_check.WAV2LIP_GAN_MODEL_PATH)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def load_gfpgan_model(self):
        
        print(f"Load GFPGAN checkpoint from: {file_check.GFPGAN_MODEL_PATH}")
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
        print(f"Load CodeFormer checkpoint from: {file_check.CODEFORMERS_MODEL_PATH}")
        
        model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(self.device)

        ckpt_path = file_check.CODEFORMERS_MODEL_PATH
        checkpoint = torch.load(ckpt_path)['params_ema']
        model.load_state_dict(checkpoint)
        return model.eval()
    
    def restore_background(self, background, model_name, tile, outscale=1.0, half=False):
        bgupsampler = self.load_realesrgan_model(model_name, tile, half=False)
        if bgupsampler is not None:
            background = bgupsampler.enhance(background, outscale=outscale)
        return background
    
    def restore_wGFPGAN(self, dubbed_face):
        dubbed_face = cv2.resize(dubbed_face.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_LANCZOS4)
        dubbed_face_t = img2tensor(dubbed_face, bgr2rgb=True, float32=True)
        normalize(dubbed_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        dubbed_face_t = dubbed_face_t.unsqueeze(0).to(self.device)
        
        try:
            output = self.restorer(dubbed_face_t, return_rgb=False, weight=self.weight)[0]
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = tensor2img(dubbed_face_t.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype(np.uint8)
        return restored_face
    
    def restore_wCodeFormer(self, dubbed_face):
        dubbed_face = cv2.resize(dubbed_face.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_LANCZOS4)
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
            print(f'\tFailed inference for CodeFormer: {error}.')
            restored_face = tensor2img(dubbed_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype(np.uint8)
        return restored_face