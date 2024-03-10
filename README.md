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
  
      

