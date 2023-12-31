<div style="display: flex; justify-content: space-between;">
    <img src = "https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/cca4987a-18e5-49f1-9eba-710f3c0655a7">
</div>

---

<div style="display: flex; justify-content: flex-start; margin-top:20px;">
    <img src="https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/84e0a59e-84c5-476c-9c20-c717b3519cf6">
    <img src="https://img.shields.io/github/forks/pawansharmaaaa/Lip_Wise?style=social">
    <img src="https://img.shields.io/github/stars/pawansharmaaaa/Lip_Wise?style=social">
    <img src="https://img.shields.io/github/watchers/pawansharmaaaa/Lip_Wise?style=social">
    <img src="https://img.shields.io/github/contributors/pawansharmaaaa/Lip_Wise?style=social&logo=github">
    <img src="https://img.shields.io/github/commit-activity/w/pawansharmaaaa/Lip_Wise?style=social&logo=github">
</div>

---

## **Introduction**

LipWise is a powerful video dubbing tool that leverages optimized inference for Wav2Lip, a cutting-edge deep learning model dedicated to generating lip-synced videos. It functions by carefully processing an input audio clip alongside a reference video featuring a speaker. This process utilizes the advanced face restoration capabilities of state-of-the-art models like GFPGAN and CodeFormer. These sophisticated models seamlessly integrate the new audio with the lip movements of the reference video, resulting in a stunningly natural and realistic final output.

* **Face Restoration Empowered by CodeFormer or GFPGAN:**
    * Streamlined inference through the elimination of redundant processes.
    * Enhanced efficiency with multi-threading implemented for the majority of preprocessing steps.
* **Unrestricted Video Compatibility:**
    * The limitation of requiring a face in every frame of the video has been lifted, allowing for greater versatility.  
* **Enhanced face detection using Mediapipe:**
    * masks generated using facial landmarks, leading to superior pasting results.
    * Facial landmarks are meticulously stored as npy files, conserving processing resources when utilizing the same video repeatedly.
* **Effortless Setup:**
    *  With the exception of manual CUDA installation, the setup process is remarkably seamless, as outlined below.

> [!IMPORTANT]
> Please Help by starring the repo. :grin:

---

# :eyeglasses: **INSTALLATION**

### :softball: **TRIAL**
<a href='https://colab.research.google.com/drive/1RSqHSi-ufSQCOlBGxCr8WOma1ihJuX9I?usp=sharing' target="_blank"><img alt='Open in Google Colab' src='https://img.shields.io/badge/OPEN_IN COLAB-100000?style=for-the-badge&logo=Google Colab&logoColor=927123&labelColor=black&color=ffffff'/></a>
> [!TIP]
> Use GPU runtime for faster processing. 

---

### :cd: **SETUP AND INFERENCE**
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
> * Clone this repository:
>   * `git clone https://github.com/pawansharmaaaa/Lip_Wise`
> * Install `Python > 3.10` from [Official Site](https://www.python.org/downloads/) or From Microsoft store.
> * Install winget from [Microsoft Store.](https://www.microsoft.com/p/app-installer/9nblggh4nns1#activetab=pivot:overviewtab)
> * Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) that is compatible with your system. The latest version generally supports most NVIDIA 10-series graphics cards and newer models.
> * Run `setup.bat`
> * Run `launch.bat`

![Debian](https://img.shields.io/badge/Debian-D70A53?style=for-the-badge&logo=debian&logoColor=white)![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)![Pop!\_OS](https://img.shields.io/badge/Pop!_OS-48B9C7?style=for-the-badge&logo=Pop!_OS&logoColor=white)![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
> * Clone this repository:
>   * `git clone https://github.com/pawansharmaaaa/Lip_Wise`
> * Make sure `python --version` is `>3.10`
> * Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) that is compatible with your system. The latest version generally supports most NVIDIA 10-series graphics cards and newer models.
> * Make `setup.sh` an executable
>   * `chmod +x ./setup.sh`
> * Run `setup.sh` by double clicking on it.
> * Make `launch.sh` an executable
>   * `chmod +x ./launch.sh`
> * Run `launch.sh` by double clicking on it.

---

## :memo: **TO-DO** List:

#### URGENT REQUIREMENTS
- [x] setup.bat / setup.sh
    - [x] create venv
    - [x] install requirements inside venv
- [x] CodeFormer arch initialization
- [ ] Documentation

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
- [ ] Inference without restorer
- [ ] Model Improvement
- [ ] Implement no_face_filter too

#### FUTURE PLANS
- [ ] Face and Audio wise Lipsync using face recognition.
- [ ] A separate tab for TTS.

#### COLAB NOTEBOOK
- [x] Optimize Inference.
- [x] Implement Checks.

---

## :hugs: ACKNOWLEDGEMENTS:

#### Thanks to the following open-source projects:
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

* <a href="https://github.com/Rudrabha/Wav2Lip" target="_blank">Wav2Lip</a>
* <a href="https://github.com/sczhou/CodeFormer" target="_blank">CodeFormer</a>
* <a href="https://github.com/TencentARC/GFPGAN" target="_blank">GFPGAN</a>
* <a href="https://github.com/googlesamples/mediapipe" target="_blank">MediaPipe</a>
