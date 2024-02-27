<p align="center" style="padding: 1rem;">
    <img style="width: 30rem;" src= "https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/5bc1b8af-879a-414b-b54a-db605a53c8f7"><img>
</p>
<div align="center" style="padding: 1rem;">
    <img src="https://img.shields.io/github/forks/pawansharmaaaa/Lip_Wise?style=social" style="padding: 0.3rem;">
    <img src="https://img.shields.io/github/stars/pawansharmaaaa/Lip_Wise?style=social" style="padding: 0.3rem;">
    <img src="https://img.shields.io/github/watchers/pawansharmaaaa/Lip_Wise?style=social" style="padding: 0.3rem;">
    <img src="https://img.shields.io/github/contributors/pawansharmaaaa/Lip_Wise?style=social&logo=github" style="padding: 0.3rem;">
    <img src="https://img.shields.io/github/commit-activity/w/pawansharmaaaa/Lip_Wise?style=social&logo=github" style="padding: 0.3rem;">
</div>

<h1 align="center" style="padding: 1rem;">
    <a href='https://colab.research.google.com/drive/1RSqHSi-ufSQCOlBGxCr8WOma1ihJuX9I?usp=sharing' target="_blank"><img alt='Open in Google Colab' src='https://img.shields.io/badge/OPEN_IN COLAB-100000?style=for-the-badge&logo=Google Colab&logoColor=927123&labelColor=black&color=ffffff'/></a>
</h1>

> [!IMPORTANT]
> Please Help by starring the repo. :grin:

---

Lip-Wise leverages Wav2Lip for audio-to-lip generation, seamlessly integrating with cutting-edge face restoration models (CodeFormer, GFPGAN, RestoreFormer) for added realism. MediaPipe ensures precise facial landmark detection, while RealESRGAN enhances background quality. Simply provide an audio clip and a reference video, and Lip-Wise orchestrates the process to deliver stunning results

**Here's what makes Lip-Wise stand out:**

- **Effortless Workflow**: Unleash your creativity with an intuitive and user-friendly interface.
- **Unleash Your Vision**: No more limitations - use any video, even those without a face in every frame.
- **Precision Meets Efficiency**: Combining enhanced face detection, landmark recognition, and streamlined processing delivers superior results with significantly faster performance.
- **Simplified Setup**: Get started quickly with minimal technical hassle - a breeze even for beginners.

---

# :eyeglasses: **INSTALLATION**
### :softball: **GETTING STARTED** 
<div align="center">
    <a href='https://colab.research.google.com/drive/1RSqHSi-ufSQCOlBGxCr8WOma1ihJuX9I?usp=sharing' target="_blank"><img alt='Open in Google Colab' src='https://img.shields.io/badge/OPEN_IN COLAB-100000?style=for-the-badge&logo=Google Colab&logoColor=927123&labelColor=black&color=ffffff'/></a>
</div>

<div align="center">
    <p style="padding: 0.5rem;">💡Tip: Make sure to use GPU runtime for faster processing.</p>
</div>

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

---

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

# **FEATURES**
LipWise empowers you to create stunningly realistic and natural results, combining the power of AI with user-friendly features:

Media Versatility:

Process both images and videos: Breathe life into your visuals, regardless of format.
Advanced image and video preprocessing: Ensure optimal quality for exceptional results.
Cutting-edge Restoration:

Harness the power of leading models: GFPGAN, RestoreFormer, and CodeFormer work in tandem to deliver exceptional detail and clarity.
RealESRGAN integration: Enhance the background quality of your visuals effortlessly.
Performance Boost:

Faster inference: Experience lightning-fast processing times for a seamless workflow.
Image Processing:

3D alignment in process image: Achieve unparalleled realism with precise facial landmark detection.
Video Processing (Coming Soon!):

No need for face in every frame: LipWise intelligently interpolates missing frames, ensuring smooth transitions and realistic lip movements.
Fast inference: Enjoy a fluid experience with rapid video processing.
Video looping: Create seamless looping videos with consistent results.
RealESRGAN integration: Elevate the background quality of your videos effortlessly

## :hugs: ACKNOWLEDGEMENTS:

#### Thanks to the following open-source projects:
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

* <a href="https://github.com/Rudrabha/Wav2Lip" target="_blank">Wav2Lip</a>
* <a href="https://github.com/sczhou/CodeFormer" target="_blank">CodeFormer</a>
* <a href="https://github.com/TencentARC/GFPGAN" target="_blank">GFPGAN</a>
* <a href="https://github.com/googlesamples/mediapipe" target="_blank">MediaPipe</a>
