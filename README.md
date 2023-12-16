# **LIP_WISE_GFPGAN**: *Wise enhancements for wav2lip.*

### :memo: <font color='red'>**Note:**</font>
#### <font color='green'>This project is actively under development, and significant improvements are expected in the near future. As I am still learning the intricacies of GitHub, I am currently committing changes on a gradual basis. To ensure the latest updates are readily accessible, I am maintaining the "clean" branch as the central repository of progress.</font>

### **Introduction**

LipWise is an enhanced version of the Wav2Lip model, a deep learning model for lip-synced video generation. It operates by processing an input audio clip alongside a reference video featuring a speaking individual. The model then synthesizes a new video where the lip movements of the person in the reference video align seamlessly with the provided audio. LipWise offers several significant improvements over the original Wav2Lip model, including:

* High-resolution output via GFPGAN and RealESRGAN
* Frame removal script
* Enhanced face detection algorithm

### **Features**

> * **High-resolution output via GFPGAN and RealESRGAN:** LipWise integrates GFPGAN and RealESRGAN, two cutting-edge GANs that specialize in upscaling images to higher resolutions while preserving impeccable image quality and intricate details. This allows LipWise to generate high-resolution lip-synced videos, resulting in a markedly clearer and more lifelike output.
> * **Frame removal script:** LipWise includes a script to automatically eliminate frames devoid of detectable facial features. This significantly enhances the overall quality of the generated videos by removing frames in which facial recognition is absent or suboptimal. This ensures that the lip-syncing process concentrates on frames characterized by precise facial data, thereby yielding a smoother and more coherent final video.
> * **Enhanced face detection algorithm:** LipWise boasts an improved face detection algorithm, augmenting the accuracy of face identification and tracking in the reference video. This heightened precision in face detection plays a pivotal role in achieving meticulous lip synchronization, ensuring the model aligns the generated lips accurately with the speaker's movements.

# :memo: **TO-DO** List:

### PREPROCESS
- [ ] Add directory check in inference in the beginning.
- [x] Make preprocessing optimal.
- [x] Clear ram after no_face_filter.
- [ ] Make face coordinates reusable:
    - [x] Saving facial coordinates as .npy file.
    - [ ] Alter code to also include eye coordinates.

### IMPROVING GAN UPSCALING
- [ ] Merge Data Pipeline with preprocessor:
    - [ ] Remove need to recrop, realign and rewarp the image.

### IMPROVING WAV2LIP
- [ ] Merge all data Pipeline:
    - [ ] Remove the need to recrop, realign, renormalizing etc.
    - [ ] Devise a way to keep frames without face in the video.
        - [ ] Understand Mels and working of wav2lip model.

### OPTIONAL
- [ ] Gradio UI
    - [ ] A tab for configuration variables.
    - [ ] A tab for Video, Audio and Output.

### FUTURE PLANS
- [ ] Face and Audio wise Lipsync using face recognition.
- [ ] A separate tab for TTS.

### COLAB NOTEBOOK
- [ ] Optimize Inference.
- [ ] Implement Checks.

## :hugs: ACKNOWLEDGEMENTS:
Thanks to the following open-source projects:
* <a href="https://github.com/Rudrabha/Wav2Lip" target="_blank">Wav2Lip</a>
* <a href="https://github.com/TencentARC/GFPGAN" target="_blank">GFPGAN</a>
* <a href="https://github.com/ShiqiYu/libfacedetection" target="_blank">libfacedetection</a>
