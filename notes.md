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