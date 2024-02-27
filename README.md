<div style="display: flex; justify-content: space-between;">
    <img src= "https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/5bc1b8af-879a-414b-b54a-db605a53c8f7"><img>
</div>

> [!IMPORTANT]
> Please Help by starring the repo. :grin:

---

<header style="display: flex; align-items: center; justify-content: space-between; background-color: #1E2533FF; margin-bottom: 0px; padding: 1rem 1rem; border-radius: 8px 8px 8px 8px;">
    <div style="display: inline-block; gap: 0rem;">
        <h1 style="background-color: black; color: white; border-radius: 8px 8px 0px 0px; font-size: 1rem; font-weight: 700; padding: 0.5rem; text-align: left; margin-bottom: 0.5em; text-shadow: hsla(0, 2%, 9%, 0.747) 0 1px 30px;">GITHUB</h1>
        <h2 style="background-color: white; color: black; border-radius: 0px 0px 8px 8px; font-size: 0.5rem; font-weight: 700; text-align: center; margin-bottom: 1em; padding: 0.2rem; text-shadow: hsla(0, 2%, 9%, 0.747) 0 1px 30px;">STATS</h2>
    </div>
    <div style="display: flex; align-items: center;">
        <img src="https://img.shields.io/github/forks/pawansharmaaaa/Lip_Wise?style=social" style="max-width: 20rem; height: auto; padding: 2px; background-color: transparent;">
        <img src="https://img.shields.io/github/stars/pawansharmaaaa/Lip_Wise?style=social" style="max-width: 20rem; height: auto; padding: 2px; background-color: transparent;">
        <img src="https://img.shields.io/github/watchers/pawansharmaaaa/Lip_Wise?style=social" style="max-width: 20rem; height: auto; padding: 2px; background-color: transparent;">
        <img src="https://img.shields.io/github/contributors/pawansharmaaaa/Lip_Wise?style=social&logo=github" style="max-width: 20rem; height: auto; padding: 2px; background-color: transparent;">
        <img src="https://img.shields.io/github/commit-activity/w/pawansharmaaaa/Lip_Wise?style=social&logo=github" style="max-width: 20rem; height: auto; padding: 2px; background-color: transparent;">
    </div>
</header>

---

## **Introduction**

LipWise is a powerful Lip-Syncing tool that leverages optimized inference for Wav2Lip, a cutting-edge deep learning model dedicated to generating lip-synced videos. It functions by carefully processing an input audio clip alongside a reference video featuring a speaker. This process utilizes the advanced face restoration capabilities of state-of-the-art models like GFPGAN and CodeFormer. These sophisticated models seamlessly integrate the new audio with the lip movements of the reference video, resulting in a stunningly natural and realistic final output.

* **Face Restoration Empowered by CodeFormer or GFPGAN:**
    * Streamlined inference through the elimination of redundant processes.
    * Enhanced efficiency with multi-threading implemented for the majority of preprocessing steps.
* **Easy to use UI**
* **Unrestricted Video Compatibility:**
    * The limitation of requiring a face in every frame of the video has been lifted, allowing for greater versatility.  
* **Enhanced face detection using Mediapipe:**
    * masks generated using facial landmarks, leading to superior pasting results.
    * Facial landmarks are meticulously stored as npy files, conserving processing resources when utilizing the same video repeatedly.
* **Effortless Setup:**
    *  With the exception of manual CUDA installation, the setup process is remarkably seamless, as outlined below.

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

## :hugs: ACKNOWLEDGEMENTS:

#### Thanks to the following open-source projects:
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

* <a href="https://github.com/Rudrabha/Wav2Lip" target="_blank">Wav2Lip</a>
* <a href="https://github.com/sczhou/CodeFormer" target="_blank">CodeFormer</a>
* <a href="https://github.com/TencentARC/GFPGAN" target="_blank">GFPGAN</a>
* <a href="https://github.com/googlesamples/mediapipe" target="_blank">MediaPipe</a>
