# Introduction
MedicalGPT training medical GPT model with ChatGPT training pipeline, implemantation ofSupervised Finetuning and Reinforcement Learning.
![image](https://github.com/Scarlett-TV/8089/blob/master/img/pipe.png)
Training MedicalGPT model：

- Stage 1: SFT (Supervised Fine-tuning) has supervised fine-tuning, constructs instruction fine-tuning data sets, and performs instruction fine-tuning on the basis of pre-trained models to align instruction intentions
- Stage 3: RM (Reward Model) reward model modeling, select a pretrained reward model
- Stage 4: RL (Reinforcement Learning) is based on human feedback reinforcement learning (RLHF), using the reward model to train the SFT model, and the generation model uses rewards or penalties to update its strategy in order to generate higher quality, more in line with human preferences text

# Training Pipline
## Stage 1: Supervised Fine-tuning
Based on the [llSourcell/medllama2_7b](https://huggingface.co/llSourcell/medllama2_7b) model, the Supervised Fine-tuning model is obtained by using medical question-and-answer data for supervised fine-tuning. 
```shell
python doctor_gpt.py
```
## Stage 2: Reward Model
In principle, we can directly use human annotations to fine-tune the model with RLHF. However, due to the time limitation, we choose a pretrained reward model [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)

## Stage 3: Reinforcement Learning
The RL (Reinforcement Learning) model aims to maximize the output of the reward model. With a fine-tuned language model and reward model, the RL loop is now prepared for execution.

The process is divided into three main steps:

1. Input a prompt, and the model generates a reply.
2. Score the responses using a reward model.
3. Optimize the policy with a round of reinforcement learning using PPO, based on the score.
![image](https://github.com/Scarlett-TV/8089/blob/master/img/pipe.png)
```shell
python ppo_new.py
```
