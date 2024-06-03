from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset,DatasetDict
import torch
from sklearn.model_selection import train_test_split
from peft import LoraConfig
from trl import SFTTrainer

MAX_LEN = 512
# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype="float16", #halves the size of the mdoel
        bnb_4bit_use_double_quant=False,
    )
device_map = "auto"#{"": 0}
model = AutoModelForCausalLM.from_pretrained("llSourcell/medllama2_7b",
                                             quantization_config=bnb_config,
                                             device_map=device_map,
                                             # use_auth_token=True
                                             )
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained("llSourcell/medllama2_7b")
tokenizer.pad_token = tokenizer.eos_token
#
## Inference
def medllama_inference(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    # Using greedy decoding and reducing the maximum length
    output = model.generate(input_ids, max_length=500)
    return tokenizer.decode(output[0])


## Process Data

data = pd.read_csv("dws_medicalqa_202404121511.csv",sep="^")
# print(data["question_en"][1])

df = data.loc[data["original_language"]=="EN"]
df = df[["question_en","answer_en"]]
tdf, vdf = train_test_split(df, test_size=0.2)
vdf = vdf.reset_index(drop=True)

tds = Dataset.from_pandas(tdf)
vds = Dataset.from_pandas(vdf)

ds = DatasetDict()
ds['train'] = tds
ds['validation'] = vds

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["question_en"], max_length=MAX_LEN,
                              truncation=True, padding='max_length',add_special_tokens=True)
    sample["labels"] = tokenizer.encode(sample["answer_en"], max_length=MAX_LEN,
                              truncation=True, padding='max_length',add_special_tokens=True)
    return sample
ds = ds.map(tokenize,remove_columns=["question_en", "answer_en"])
ds.set_format(type="torch",columns=["input_ids", "labels"])

## Training Process


training_arguments = TrainingArguments(
    output_dir='results/',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim='paged_adamw_32bit',
    save_steps=5000,#5000

    logging_dir='out_logs/',
    logging_steps=2,

    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=5000, #5000
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant',
)
model.config.use_cache = False

model.config.pretraining_tp = 1
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


# Define data collator to handle tokenization and collation
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training
torch.cuda.empty_cache()
trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    peft_config=peft_config,
    dataset_text_field="input",
    max_seq_length=MAX_LEN, #512
    args=training_arguments,
    data_collator=data_collator,
    packing=False,
)
trainer.train()
trainer.save_model("output_model")

log_result = pd.DataFrame(trainer.state.log_history)
log_result.to_json(path_or_buf="/home/Data1/Medical-Public/files/dev/OralQALLM/python/finetune/log_out.json",orient="records")


