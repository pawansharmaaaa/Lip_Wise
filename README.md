<p align="center" style="padding: 1rem;">
    <img style="width: 30rem;" src= "https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/5bc1b8af-879a-414b-b54a-db605a53c8f7"><img>
</p>
<div align="center" style="padding: 1rem;">
    <img src="https://img.shields.io/github/stars/pawansharmaaaa/Lip_Wise?style=social" style="padding: 0.3rem;">
    <img src="https://img.shields.io/github/forks/pawansharmaaaa/Lip_Wise?style=social" style="padding: 0.3rem;">
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

**Lip-Wise leverages Wav2Lip for audio-to-lip generation, seamlessly integrating with cutting-edge face restoration models (CodeFormer, GFPGAN, RestoreFormer) for added realism. MediaPipe ensures precise facial landmark detection, while RealESRGAN enhances background quality. Simply provide an audio clip and a reference video, and Lip-Wise orchestrates the process to deliver stunning results.**

**Here's what makes Lip-Wise stand out:**

- **Effortless Workflow**: Unleash your creativity with an intuitive and user-friendly interface.
- **Unleash Your Vision**: No more limitations - use any video, even those without a face in every frame.
- **Precision Meets Efficiency**: Combining enhanced face detection, landmark recognition, and streamlined processing delivers superior results with significantly faster performance.
- **Simplified Setup**: Get started quickly with minimal technical hassle - a breeze even for beginners.

---

# :framed_picture: UI Screenshots:
<p align="center" style="padding: 1rem;">
    <img height="300" padding="1rem" src= "https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/c69c1973-5350-4b1b-a6d7-1c00212d1757"></img>
    <img height="300" padding="1rem" src= "https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/87749293-2bdf-48fa-8744-c3a4fdf72925"></img>
</p>

# :eyeglasses: **INSTALLATION**
### **:zap: QUICK INFERENCE**
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

# :control_knobs: **FEATURES**

**LipWise empowers you to create stunningly realistic and natural results, combining the power of AI with user-friendly features:**

#### **Media Versatility:**

- **Process both images and videos:** Breathe life into your visuals, regardless of format.
- **Advanced image and video preprocessing:** Ensure optimal quality for exceptional results.

#### **Cutting-edge Restoration:**

- **Harness the power of leading models:** GFPGAN, RestoreFormer, and CodeFormer work in tandem to deliver exceptional detail and clarity.
- **RealESRGAN integration:** Enhance the background quality of your visuals effortlessly.

#### **Image Processing:**

- **3D alignment in process image:** Achieve unparalleled realism with precise facial landmark detection.

#### **Video Processing:**

- **No need for face in every frame:** LipWise intelligently interpolates missing frames, ensuring smooth transitions and realistic lip movements.
- **Fast inference:** Enjoy a fluid experience with rapid video processing.
- **Video looping:** Create seamless looping videos with consistent results.
- **RealESRGAN integration:** Elevate the background quality of your videos effortlessly

## :scroll: LICENSE AND ACKNOWLEDGEMENT

Lip-Wise is released under Apache License Version 2.0.


<div align="left">
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" height="30" alt="PyTorch">
    </a>
    <a href="https://numpy.org/">
        <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" height="30" alt="NumPy">
    </a>
    <a href="https://opencv.org/">
        <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" height="30" alt="OpenCV">
    </a>
    <a href="https://github.com/googlesamples/mediapipe">
        <img src="https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/5d8d14f3-c12d-431c-a7da-bc08ca823147" height="30" alt="MediaPipe">
    </a>
</div>

### Citations
* <a href="https://github.com/Rudrabha/Wav2Lip" target="_blank">Wav2Lip</a>
```
    @inproceedings{10.1145/3394171.3413532,
        author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
        title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
        year = {2020},
        isbn = {9781450379885},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3394171.3413532},
        doi = {10.1145/3394171.3413532},
        booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
        pages = {484–492},
        numpages = {9},
        keywords = {lip sync, talking face generation, video generation},
        location = {Seattle, WA, USA},
        series = {MM '20}
    }
```
* <a href="https://github.com/sczhou/CodeFormer" target="_blank">CodeFormer</a>
```
    @inproceedings{zhou2022codeformer,
        author = {Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
        title = {Towards Robust Blind Face Restoration with Codebook Lookup TransFormer},
        booktitle = {NeurIPS},
        year = {2022}
    }
```
* <a href="https://github.com/TencentARC/GFPGAN" target="_blank">GFPGAN and RestoreFormer</a>
```
    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }
```
* <a href="https://github.com/xinntao/Real-ESRGAN" target="_blank">RealESRGAN</a>
```
    @InProceedings{wang2021realesrgan,
        author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }
```

<!-- ### :hugs: Thanks to the following open-source projects: -->

## :e-mail: Contact

Reach out to me @ `lipwisedev@gmail.com`
