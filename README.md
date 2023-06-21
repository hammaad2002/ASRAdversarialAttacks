# ASR-Adversarial-Attacks

Welcome to the ASR Adversarial Attacks repository! This is a collection of adversarial attacks for Automatic Speech Recognition (ASR) systems. The attacks in this 'main' branch are specifically designed for the wav2vec2 model from the Torchaudio hub. Additionally, I have implemented a separate branch called 'huggingface' for the Hugging Face version of this repository. This is for those who want to perform these attacks on the ASR models available on their platform.

This repository contains a collection of adversarial attacks for Automatic Speech Recognition (ASR) systems. The repository includes a file that implements several popular attack methods, including the Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Projected Gradient Descent (PGD), Carlini and Wagner (CW), and Imperceptible CW.

Most of these attacks are designed to generate perturbations in the audio signal that are imperceptible or quasi-imperceptible to the human ear, but at the same time causes ASR systems to produce incorrect transcriptions. The implementation of these attacks in this repository can be used to evaluate the robustness of ASR models and to develop defenses against such attacks.

PLEASE STAR THE REPOSITORY IF YOU FIND IT INTERESTING OR HELPFUL!
