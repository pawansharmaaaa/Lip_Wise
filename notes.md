# :memo: **TO-DO** List:

### PREPROCESS
- [x] Add directory check in inference in the beginning.
- [x] Make preprocessing optimal.
- [x] Clear ram after no_face_filter.
- [x] Make face coordinates reusable:
    - [x] Saving facial coordinates as .npy file.
    - [x] Alter code to also include eye coordinates.

### IMPROVING GAN UPSCALING
- [x] Merge Data Pipeline with preprocessor:
    - [x] Remove need to recrop, realign and rewarp the image.

### IMPROVING WAV2LIP
- [x] Merge all data Pipeline:
    - [x] Remove the need to recrop, realign, renormalizing etc.
    - [x] Devise a way to keep frames without face in the video.
        - [x] Understand Mels and working of wav2lip model.

### OPTIONAL
- [ ] Gradio UI
    - [ ] A tab for configuration variables.
    - [ ] A tab for Video, Audio and Output.
    - [x] A tab for Image, Audio and output.

### FUTURE PLANS
- [ ] Face and Audio wise Lipsync using face recognition.
- [ ] A separate tab for TTS.

### COLAB NOTEBOOK
- [ ] Optimize Inference.
- [ ] Implement Checks.