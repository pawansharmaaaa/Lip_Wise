import file_check
import torch

from models import Wav2Lip
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load(checkpoint_path):
	if DEVICE == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_wav2lip_model():
    model = Wav2Lip()
    print(f"Load checkpoint from: {file_check.WAV2LIP_MODEL_PATH}")
    checkpoint = _load(file_check.WAV2LIP_MODEL_PATH)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(DEVICE)
    return model.eval()

def load_gfpgan_model():

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

    return gfpgan.eval()