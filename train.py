from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np 
import pandas as pd
import os
import pickle

#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
i = 6
checkpoint = 0
#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


def train_model(
    dataset, model, tokenizer, checkpoint,
    batch_size=32, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=True,
):
    acc_steps = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #input = data.to(device)
    #model = MyModule(...).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, 1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None
    for epoch in range(checkpoint,5):
#    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model

import torch
from torch.utils.data import Dataset, DataLoader

class Descriptions(Dataset):  
    def __init__(self, df, price, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.descriptions = []

        for index, row in df.iterrows():
          self.descriptions.append(torch.tensor(
                self.tokenizer.encode(f"<|{price}|><|{row.title} {row.feature}|>{row.description[:max_length]}<|endoftext|>")
            ))               
        if truncate:
            self.descriptions = self.descriptions[:20000]
        
    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        return self.descriptions[item]
    
    def add(self, other_descriptions):
        self.descriptions = self.descriptions + other_descriptions
    
    def return_descriptions(self):
        return self.descriptions

expensive = pd.read_csv('expensive.csv')
cheap = pd.read_csv('cheap.csv')

train1, train2, train3, train4, train5, train6, train7, validate, test = \
              np.split(expensive.sample(frac=1, random_state=1), 
                       [int(.1*len(expensive)), int(.2*len(expensive)), int(.3*len(expensive)), int(.4*len(expensive)), int(.5*len(expensive)), int(.6*len(expensive)), int(.7*len(expensive)), int(.85*len(expensive))])

trains = [train1, train2, train3, train4, train5, train6, train7]

train21, train22, train23, train24, train25, train26, train27, validate2, test2 = \
              np.split(cheap.sample(frac=1, random_state=1), 
                       [int(.1*len(cheap)), int(.2*len(cheap)), int(.3*len(cheap)), int(.4*len(cheap)), int(.5*len(cheap)), int(.6*len(cheap)), int(.7*len(cheap)), int(.85*len(cheap))])              
#dataset = Descriptions(train, "expensive", truncate=True, gpt2_type="gpt2")   

train2s = [train21, train22, train23, train24, train25, train26, train27]

#dataset2 = Descriptions(train2, "cheap", truncate=True, gpt2_type="gpt2")  
#dataset.add(dataset2.return_descriptions())

#for i in range(7):
#	dataset = Descriptions(trains[i], "expensive", truncate=True, gpt2_type="gpt2")
#	dataset2 = Descriptions(train2s[i], "cheap", truncate=True, gpt2_type="gpt2")
#	dataset.add(dataset2.return_descriptions())
#	if i == 0:
#		model = GPT2LMHeadModel.from_pretrained('gpt2')
#	else:
#		model = pickle.load(open('langgen_model.pkl', 'rb'))
#
#	finetuned_model = train_model(dataset, model, tokenizer)
#	pickle.dump(finetuned_model, open('langgen_model.pkl', 'wb'))

dataset = Descriptions(trains[i], "expensive", truncate=True, gpt2_type="gpt2")
dataset2 = Descriptions(train2s[i], "cheap", truncate=True, gpt2_type="gpt2")
dataset.add(dataset2.return_descriptions())
#model = GPT2LMHeadModel.from_pretrained('gpt2')

model = pickle.load(open('langgen_model.pkl', 'rb'))
if checkpoint > 0:
	model.load_state_dict(torch.load('wreckgar-' + str(checkpoint-1) + '.pt'))
finetuned_model = train_model(dataset, model, tokenizer,checkpoint)
pickle.dump(finetuned_model, open('langgen_model.pkl', 'wb'))
