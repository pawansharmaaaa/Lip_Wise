## :memo: **TO-DO** List:

#### URGENT REQUIREMENTS
- [x] Change mask in seamless clone and give it a try
- [x] setup.bat / setup.sh
    - [x] create venv
    - [x] install requirements inside venv
- [x] CodeFormer arch initialization
- [x] Documentation

#### PREPROCESS
- [x] Add directory check in inference in the beginning.
- [x] Make preprocessing optimal.
- [x] Clear ram after no_face_filter.
- [x] Make face coordinates reusable:
    - [x] Saving facial coordinates as .npy file.
    - [x] Alter code to also include eye coordinates.

#### IMPROVING GAN UPSCALING
- [x] Merge Data Pipeline with preprocessor:
    - [x] Remove need to recrop, realign and rewarp the image.

#### IMPROVING WAV2LIP
- [x] Merge all data Pipeline:
    - [x] Remove the need to recrop, realign, renormalizing etc.
    - [x] Devise a way to keep frames without face in the video.
        - [x] Understand Mels and working of wav2lip model.

#### OPTIONAL
- [x] Gradio UI
    - [x] A tab for Video, Audio and Output.
    - [x] A tab for Image, Audio and output.

#### FURTHER IMPROVEMENTS
- [x] Inference without restorer
- [ ] Model Improvement
- [ ] Implement no_face_filter too

#### COLAB NOTEBOOK
- [x] Make it intuitive with proper instructions.
- [x] Optimize Inference.
- [x] Implement Checks.

#### FUTURE PLANS
- [ ] Face and Audio wise Lipsync using face recognition.
- [ ] A separate tab for TTS.

---