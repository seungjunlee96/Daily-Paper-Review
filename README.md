# A Survey on GANs for Anomaly Detection
Brief review of GANs for Anomaly Detection.<br>
Paper Link : https://arxiv.org/pdf/1906.11632.pdf

## What is "Anomaly"?
- the pattern in data that do not conform to a well-defined notion of normal behavior

## What is GAN (Generative Adversarial Network)?
GAN (Generative Adversarial Network) : learns a generator that maps samples from an arbitrary latent distribution (noise prior) to data as well as a discriminator which tries to distinguish between real and generated samples.

GAN in anomaly detection
- GANs model complex and high dimensional **"distribution"** of real-world data.
- [Adversarial Feature Learning](https://arxiv.org/abs/1605.09782) : built the basis of GAN based anomaly detection approaches where the **BiGAN**(Bidirectional GAN) architecture has been proposed.


# BiGAN (Bidirectional GAN)
- inverse of the generator *E* that learns a mapping from data to latent space : (**x**, *E*(**x**))
- (*G*(z) ,z) 

# AnoGAN
- Training a GAN on normal samples only, makes the generator learn the manifold *X* of normal samples.
- To find a point z in the latent space that corresponds to a generated value G(z) that is simialr to the query value x located on the manifold X of the positive samples.
- Total loss is defined as the weighted sum of (1) residual loss and (2) discrimination loss.

## pros
- Showed that GANs can be used for anomaly detection.
- New mapping scheme from latent space to input data space.
- Used the same mapping scheme to define an anomaly score.<br>

## cons
- Requires optimization steps for every new input : bad test-time performance
- The GAN objective has not been modified to take into account the need for the inverse mapping learning.
- The anomaly score is difficult to interpret, not being in the probability range.

# EGBAD
- learning an encoder *E* able to map input samples to their latent representation during the adversarial training.
- Allow computing the anomaly score without optimization steps during the inference as it happens in AnoGAN.

# GANomaly
## Generator network
- three elements in series
- an encoder G_E a decoder G_D (both assembling an autoencoder structure) and another encoder E

## Discriminator Network

## Loss functions
- Adversarial Loss : Feature Matching Loss
- Contextual Loss : L1 Loss 
- Encoder Loss : let the generator network learn how to best encode a normal image

## Pros
- An encoder is learned during the training process, hence we don't have the need for a research process as in AnoGAN
- Using an autoencoder like structure (no use of noise prior) makes the entire learning process faster
- The anomaly score is easier to interpret
- The contextual loss can be used to localize the anomaly

## Cons
- It allows to detect anomalies both in the image and latent space, but the results couldn't match: a higher anomaly score, that's computed only in the latent space, can be associated with a generated sample with a low contextual loss value an thus very similar to the input - and vice versa.
- Defines a new anomaly score




