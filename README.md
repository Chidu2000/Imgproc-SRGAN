                                               # SRGAN (Super-Resolution Generative Adversarial Network)


Overview

SRGAN is a deep learning model designed for single-image super-resolution, which aims to generate high-resolution images from low-resolution inputs. This repository contains the implementation of SRGAN using the PyTorch framework.




Features

  Single-image super-resolution using Generative Adversarial Networks (GANs)
  Implemented using PyTorch for flexibility and scalability
  Provides both the generator and discriminator networks for training and evaluation
  Pre-trained models for quick testing and deployment
  Configurable training parameters and options for experimentation




Installation

1. Clone the repository:
   git clone https://github.com/your_username/srgan.git
   cd srgan
2. Dependencies:
   For **conda** environment setup:
   Open anaconda prompt and cd to the folder where you have your environment.yml file
   conda env create -f environment.yml
   conda activate srganenv_gpu

   Install pytorch as per your resources
   #### GPU
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

   ##### CPU Only
    conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch




Training and Testing   

Training
  To train the SRGAN model, run the following command:

  !python main.py --LR_path custom_dataset_cars/hr_train_LR --GT_path custom_dataset_cars/hr_train_HR

Test your Model:
  !python main.py --mode test_only --LR_path test_data/cars --generator_path ./model/srgan_custom.pt




Pre-trained Models

Pre-trained SRGAN models are available in the pretrained_models directory. These models can be used for quick testing and deployment.




Modes

train: This mode is used for training the model
test: Test mode involves both evaluating the model's performance and potentially making predictions on unseen data.
test-only: It is essentially the deployment phase where the model is deployed into a production environment or used for real-world applications.

### Dataset Description

## Dataset 1 - Referred in Paper
In this project, we utilize three benchmark datasets for evaluating the performance of our image processing algorithms: Set5, Set14, and BSD100. Additionally, we employ the testing set of BSD300 for further validation. Below is a detailed description of each dataset:

1. **Set5**: This dataset is a popular choice for benchmarking image super-resolution and other image processing tasks. It consists of five high-resolution images, each representing different types of scenes and textures. The images are often used to evaluate the quality of upscaling algorithms by comparing the reconstructed high-resolution images with the original ones.

2. **Set14**: Similar to Set5, Set14 is another widely-used dataset for image processing evaluations, particularly in super-resolution. It contains 14 high-resolution images, covering a broader range of scenes and complexities. The diversity in this dataset helps in assessing the robustness and generalizability of image processing algorithms.

3. **BSD100**: Part of the Berkeley Segmentation Dataset (BSDS), BSD100 includes 100 natural images for testing image segmentation, denoising, and super-resolution algorithms. The images are carefully selected to represent various real-world scenarios, making it a challenging dataset for evaluating the performance of image processing techniques.

4. **Testing Set of BSD300**: The BSD300 dataset is an extension of BSD100, consisting of 300 images in total. For our purposes, we use the testing set of BSD300, which comprises 200 images not included in BSD100. This set provides a larger and more diverse collection of images for comprehensive testing and validation of our algorithms.

## Dataset 2
### Dataset Description

We utilize the DIV2K dataset for evaluating the performance of our image processing algorithms. Below is a detailed description of the dataset:

**DIV2K Dataset**: DIV2K is a high-quality image dataset specifically designed for the task of image super-resolution and other related image processing challenges. It consists of 1,000 diverse, high-resolution images (2K resolution) that are divided into three subsets: a training set with 800 images, a validation set with 100 images, and a test set with another 100 images. The images in the DIV2K dataset cover a wide range of scenes, including natural landscapes, urban environments, and intricate textures, making it a comprehensive benchmark for evaluating super-resolution algorithms.

The DIV2K dataset is renowned for its high-quality images and diversity, providing a robust platform for developing and testing advanced image processing techniques. By using this dataset, researchers and practitioners can assess the effectiveness of their algorithms in enhancing image resolution while maintaining or improving image fidelity. The results obtained on the DIV2K dataset are often considered as a standard benchmark in the field of image super-resolution.

In our research, the DIV2K dataset serves as a crucial tool for validating the performance of our proposed image processing algorithms. By comparing the results on this dataset with those obtained from other benchmarks, we can ensure the competitiveness and generalizability of our methodologies in various real-world scenarios.
