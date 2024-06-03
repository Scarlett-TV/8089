from trl import PPOTrainer,PPOConfig,AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
import json
import pandas as pd
from datasets import Dataset,DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from peft import PeftModel
# import wandb

# wandb.init(project="test_ppo")
## Load Model
model_path = "medllama2_415"#"medllama2_415"
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
MAX_LEN = 15
# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype="float16", #halves the size of the mdoel
        bnb_4bit_use_double_quant=False,
    )
device_map = {"": 0}#"auto"#{"": 0}
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,
                                             local_files_only=True,
                                             quantization_config=bnb_config,
                                             device_map=device_map)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,
                                             local_files_only=True,
                                             quantization_config=bnb_config,
                                             device_map=device_map)

# base_model = AutoModelForCausalLMWithValueHead.from_pretrained("llSourcell/medllama2_7b",
#                                              # local_files_only=True,
#                                              quantization_config=bnb_config,
#                                              device_map=device_map)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          local_files_only=True,
                                          # padding_side='left'
                                          )
# tokenizer.pad_token = tokenizer.eos_token
# Load Reward Model
rank_model, rank_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
print("Finish Loading")

## Reward Score
def reward_score(question,answer):
    inputs = rank_tokenizer(question, answer, return_tensors='pt')
    score = rank_model(**inputs).logits[0].cpu().detach()
    return(score)

## Inference
def _inference(prompt):
 input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
 # Using greedy decoding and reducing the maximum length
 output = model.generate(input_ids, max_length=512,eos_token_id=-1)
 return input_ids,output[0],tokenizer.decode(output[0])

def _ref_inference(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    # Using greedy decoding and reducing the maximum length
    output = ref_model.generate(input_ids, max_length=512, eos_token_id=-1)
    return input_ids, output[0], tokenizer.decode(output[0])
## Load Data
questions = ['What are dental implants?',
            'What does an implant consist of?',
            'How long does it take before crowns can be fitted to an implant?',
            'Are implants safe?',
            'What are the advantages of implants?',
            'Are dental implants suitable for everyone?',
            'I have periodontitis. Can I have dental implants?',
            'When are dental implants not an option?',
            'What are the stages of implant therapy?',
            'What are bone grafting and bone regeneration?',
            'How do I look after my implants properly?',
            'What are peri-implant diseases?',
            'What is the treatment for peri-implant diseases?',
            'What is periodontitis?',
            'What are the symptoms of periodontitis?',
            'What is the difference between gingivitis and periodontitis?',
            'My gums are receding: does that mean I have periodontitis?',
            'What are the causes of periodontitis?',
            'What are the risk factors for periodontitis?',
            'What can I do to prevent periodontal disease?',
            'How is periodontitis diagnosed?',
            'How is periodontitis treated?',
            'What is periodontology?',
            'What is a periodontist?',
            'What are the links between periodontitis and other diseases?']


config = PPOConfig(
    # is_peft_model=True,
    steps=10, learning_rate=1.41e-5, remove_unused_columns=False, mini_batch_size=1,batch_size=1, #log_with="wandb"
)

ppo_trainer = PPOTrainer(
    model=model,
    # ref_model=base_model,
    config=config,
    tokenizer=tokenizer,
)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens":MAX_LEN
}

epochs = 10

for i in tqdm(range(epochs),"epoch: "):
    for question in questions:
        input_ids,response_tensor,response = _inference(question)
        query_tensor = [torch.squeeze(input_ids).to("cuda:0")]
        # print(response)
        reward = [reward_score(question, response).to("cuda:0")]
        response_tensor = [response_tensor.to("cuda:0")]
        stats = ppo_trainer.step(query_tensor, response_tensor, reward)


### check_model
ppo_data = dict()
ppo_data["question"] = []
ppo_data["response_before"] = []
ppo_data["response_after"] = []
ppo_data["reward_before"] = []
ppo_data["reward_after"] = []
for q in questions:
    ppo_data["question"].append(q)
    _,_,before_response = _ref_inference(q)
    ppo_data["response_before"].append(before_response)
    _,_,after_response = _inference(q)
    ppo_data["response_after"].append((after_response))
    ppo_data["reward_before"] .append(reward_score(q,before_response).item())
    ppo_data["reward_after"].append(reward_score(q,after_response).item())

ppo_results = pd.DataFrame(ppo_data)
print(ppo_results)
with open("ppo_results.json", 'w') as fp:
    json.dump(ppo_data, fp)

#### Save model
model.save_pretrained("ppo_new")
tokenizer.save_pretrained("ppo_new")
