# ASR-Adversarial-Attacks

Welcome to the ASR Adversarial Attacks repository! This is a collection of adversarial attacks for Automatic Speech Recognition (ASR) systems. The attacks in this repository are specifically designed for the wav2vec2 model from Torchaudio hub, but I'm planning to make these attacks applicable to most of the Hugging Face ASR models in the future.

This repository contains a collection of adversarial attacks for Automatic Speech Recognition (ASR) systems. The repository includes a file that implements several popular attack methods, including Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Projected Gradient Descent (PGD), Carlini and Wagner (CW), and Imperceptible CW.

These attacks are designed to generate perturbations in the audio signal that are imperceptible to the human ear, but can cause ASR systems to produce incorrect transcriptions. The implementation of these attacks in this repository can be used to evaluate the robustness of ASR models and to develop defenses against such attacks.
