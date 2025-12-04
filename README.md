# About
This is images contrastive conditional flow-matching implementation with classifier-free guidance.

Model trains a flow matching model that maps random normal distribution to images. https://arxiv.org/abs/2210.02747
It conditions generation on CLIP embeddings of images. https://arxiv.org/abs/2103.00020
Loss functions adds constraint that for different CLIP embeddings from same prior we get different flow directions. https://arxiv.org/abs/2506.05350

# Setup
1. Pre-generate VAE latents and CLIP embeddings.
Open `prepare_dataset.ipynb`. Provide path to your images dataset and run latents generation.
This step will pre-generate image VAE latents and clip embeddings that will be used for training.
2. Open `images-latent.ipynb`
Set path to your dataset and save dir for model. Run all cells.
3. Evaluate at `sample_latents.ipynb`
When training occurs, you can use this module to check how training goes and generate images.

# Examples
On my private dataset of 2d trees assets, after 432 epochs, I got r2 ~ 0.75 with following results.
<img width="790" height="382" alt="image" src="https://github.com/user-attachments/assets/f18c80a6-fe93-4140-acb6-7e05daef2eec" />
<img width="790" height="382" alt="image" src="https://github.com/user-attachments/assets/2986de31-4560-4395-882a-4043e787c073" />

