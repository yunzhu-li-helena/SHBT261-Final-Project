# SHBT261-Final-Project
SHBT Final Proejct Code - Helena (Yun Zhu) Li and Yike Cheng

# TextVQA Prompt Engineering Final Project

This repository contains the code used for the final project on visual understanding with the TextVQA dataset. The project evaluates whether prompt engineering can improve the performance of a frozen vision-language model on TextVQA without fine-tuning model parameters.

The main experiment uses **Qwen2.5-VL-3B-Instruct** on the TextVQA validation split and compares four prompt conditions: plain, concise, OCR-aware, and strict OCR-aware prompting.

## Repository Contents

```text
.
├── README.md
├── textvqa_pipeline.py
└── colab_prompt_engineering_final.ipynb
