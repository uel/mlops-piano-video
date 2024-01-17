# Group members
- Jaykrishnan Gopalakrishna Pillai
- Filip Danielsson
- Filip Koňařík

# Goal 
Using MLOps procedures to develop a model to generate a sequence of frame of someone playing the piano.

# Frameworks
For the piano image generation we will test different generative models such as [denoising diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) or GANs (generative adversarial networks).
An example of using diffusion models for conditional image generation: https://github.com/TeaPearce/Conditional_Diffusion_MNIST


# Data
We will extract training data from publicly available videos using https://github.com/uel/BP. The generative model can be conditioned on different types of data extracted by the library for example hand placement, played notes, key location, previous frame. 

# Models
No existing models for this specific task exist at the moment, we will therefore train a generative model from scratch. The data extraction step uses deep learning models for [hand landmarking](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker), keyboard detection and [piano transcription](https://github.com/bytedance/piano_transcription).

# Code coverage
Code coverage: 44%