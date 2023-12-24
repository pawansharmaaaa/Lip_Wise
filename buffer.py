def infer_video(video_path, audio_path, max_batch_size=16):

    # Perform checks to ensure that all required files are present
    file_check.perform_check()

    # Get input type
    input_type, vid_ext = file_check.get_file_type(video_path)
    if input_type != "video":
        raise Exception("Input file is not a video. Try again with a video file.")
    
    # Get audio type
    audio_type, aud_ext = file_check.get_file_type(audio_path)
    if audio_type != "audio":
        raise Exception("Input file is not an audio.")
    if aud_ext != "wav":
        print("Audio file is not a wav file. Converting to wav...")
        # Convert audio to wav
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, os.path.join(MEDIA_DIRECTORY, 'aud_input.wav'))
        subprocess.call(command, shell=True)
        audio_path = os.path.join(MEDIA_DIRECTORY, 'aud_input.wav')
    
    # Generate audio spectrogram
    print("Generating audio spectrogram...")
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)  