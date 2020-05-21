# Wasserstein GAN
- WGAN (Wasserstein GAN using Weight Clipping) 
- WGAN-GP (Wasserstein GAN using Gradient Penalty)

paper
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

The Wasserstein GAN (WGAN) is a GAN variant which uses the 1-Wasserstein distance, rather than the JS-Divergence, to measure the difference between the model and target distributions.


## What does it mean to learn a probability distribution?
- classical answer: to learn a probability density by defining a parametric family of densities and finding the one that maximized the likelihood on our data.
- minimizing the Kullback-Leibler divergence KL(Pr||Pθ)

## Original GAN is hard to train
- Saturated Gradients
- In general, Pr and Pg are unlikely to have non-negligible intersection
- Mode collapse : During training, the generator may collapse to a setting where it always produces same outputs.
- Unstable (hard to balance between Discriminator and Generator)
- Lack of a proper evaluation metric

## Distance metrics between two distribution
- Kullback-Leibler Divergence (not distance) : A measure of how one probability distribution is different from a second, reference probability distribution. (x 를 Pr의 확률공간에서 샘플링 했을 때, 'Pr(x)의 엔트로피 값과 Pg(x)의 엔트로피 값의 차'의 기대값.)
- Jensen-Shannon Divergence (not distance) : a method of measuring the similarity between two probability distributions.
- [Total Variation Distance](https://en.wikipedn ia.org/wiki/Total_variation_distance_of_probability_measures) : Informally, it is the largest possible difference between the probabilities that the two probabilitiy distributions can assign to the same event.


## Wasserstein-1 Distance (Earth Mover's Distance)
Informally it can be interpreted as the minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution.

why Wasserstein Distance is good?
- KL divergence gives us infinity when two distributions are disjoint
- The value of Jenson-Shannon divergence has sudden jump, not differentiable at some point.


## Contributions
- An analysis of the convergence properties of the value function being optimized by GANs
- Wasserstein distance to produce a value function which has better theoretical properties than the original.
- WGAN requires the discriminator (called the *critic* in the paper) must lie within the space of **1-Lipschitz** functions, which weight clipping enforces

```python
"""clip weights of discriminator"""
for param in discriminator.parameters():
    param.data.clamp_(-clip_value, clip_value)
```

Note that, we do not apply sigmoid at the output of Discriminator as the output of Discriminator is no longer a probability.

# Improved Training of Wassertein GANs
paper 
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

Problem
- GANs suffer from **training instability**.
- The use of weight clipping in WGAN to enforce a **Lipschitz** constraint on the critic
- critic weight clipping can lead to undesired problems
- "Weight clipping is a clearly terrible way to enforce a Lipschitz constraint"

Solution : alternative cliping weights
- Penalize the norm of gradient of the critic with respect to its input.
- Gradient Penalty (WGAN-GP)
