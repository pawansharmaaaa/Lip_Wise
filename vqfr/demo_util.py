import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from vqfr.archs.vqfrv1_arch import VQFRv1
from vqfr.archs.vqfrv2_arch import VQFRv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VQFR_Demo():
    """Helper for restoration with VQFR.
    It will detect and crop faces, and then resize the faces to 512x512.
    VQFR is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.
    Args:
        model_path (str): The path to the VQFR model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The VQFR architecture. Option: original. Default: original.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='v1', bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the VQFR
        if arch == 'v1':
            self.vqfr = VQFRv1(
                base_channels=128,
                proj_patch_size=32,
                resolution_scale_rates=[1, 2, 2, 2, 2, 2],
                channel_multipliers=[1, 1, 2, 2, 2, 4],
                encoder_num_blocks=2,
                decoder_num_blocks=3,
                quant_level=['Level_32'],
                fix_keys=['embedding'],
                inpfeat_extraction_opt={
                    'in_dim': 3,
                    'out_dim': 32
                },
                align_from_patch=32,
                align_opt={
                    'Level_32': {
                        'cond_channels': 32,
                        'cond_downscale_rate': 32,
                        'deformable_groups': 4
                    },
                    'Level_16': {
                        'cond_channels': 32,
                        'cond_downscale_rate': 16,
                        'deformable_groups': 4
                    },
                    'Level_8': {
                        'cond_channels': 32,
                        'cond_downscale_rate': 8,
                        'deformable_groups': 4
                    },
                    'Level_4': {
                        'cond_channels': 32,
                        'cond_downscale_rate': 4,
                        'deformable_groups': 4
                    },
                    'Level_2': {
                        'cond_channels': 32,
                        'cond_downscale_rate': 2,
                        'deformable_groups': 4
                    },
                    'Level_1': {
                        'cond_channels': 32,
                        'cond_downscale_rate': 1,
                        'deformable_groups': 4
                    }
                },
                quantizer_opt={
                    'Level_32': {
                        'type': 'L2VectorQuantizer',
                        'in_dim': 512,
                        'num_code': 1024,
                        'code_dim': 256,
                        'reservoir_size': 16384,
                        'reestimate_iters': 2000,
                        'reestimate_maxiters': -1,
                        'warmup_iters': -1
                    }
                })
        elif arch == 'v2':
            self.vqfr = VQFRv2(
                base_channels=64,
                channel_multipliers=[1, 2, 2, 4, 4, 8],
                num_enc_blocks=2,
                use_enc_attention=True,
                num_dec_blocks=2,
                use_dec_attention=True,
                code_dim=256,
                inpfeat_dim=32,
                align_opt={
                    'cond_channels': 32,
                    'deformable_groups': 4
                },
                code_selection_mode='Predict',  # Predict/Nearest
                quantizer_opt={
                    'type': 'L2VectorQuantizer',
                    'num_code': 1024,
                    'code_dim': 256,
                    'spatial_size': [16, 16]
                })

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device)

        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.vqfr.load_state_dict(loadnet[keyname], strict=True)
        self.vqfr.eval()
        self.vqfr = self.vqfr.to(self.device)

    @torch.no_grad()
    def enhance(self, img, fidelity_ratio=None, has_aligned=False, only_center_face=False, paste_back=True):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.vqfr(cropped_face_t, fidelity_ratio=fidelity_ratio)['main_dec'][0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for VQFR: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
