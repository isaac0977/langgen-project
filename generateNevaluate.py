from awessome.awessome_builder import *
import pickle
import torch
import pandas as pd
from rouge_metric import PyRouge
import nltk
nltk.download('averaged_perceptron_tagger')
desirable = pd.read_csv('../expensive.csv')
undesirable = pd.read_csv('../cheap.csv')
model = pickle.load(open('../langgen_model.pkl', 'rb'))
roberta = pickle.load(open('../roberta.pkl', 'rb'))

avg_builder = SentimentIntensityScorerBuilder('avg', 'bert-base-nli-mean-tokens', 'euclidean', '100', True)
labmt_avg_scorer = avg_builder.build_scorer_from_prebuilt_lexicon('labmt')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

class TestDescriptions(Dataset):  
    def __init__(self, df, price, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.descriptions = []
        self.gold_label = []

        for index, row in df.iterrows():
          self.descriptions.append(f"<|{price}|><|{row.title} {row.feature}|>")
          self.gold_label.append(f"{row.description}")        
        if truncate:
            self.descriptions = self.descriptions[:20000]
            self.gold_label = self.gold_label[:20000]

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        return self.descriptions[item]
    
    def add(self, other_descriptions):
        self.descriptions = self.descriptions + other_descriptions
    
    def return_descriptions(self):
        return self.descriptions
    
    def get_original(self, item):
        return self.gold_label[item]

import numpy as np

_, _, expensive_test = \
              np.split(desirable.sample(frac=1, random_state=1), 
                       [int(.7*len(desirable)), int(.85*len(desirable))])

_, _, cheap_test = \
              np.split(undesirable.sample(frac=1, random_state=1), 
                       [int(.7*len(undesirable)), int(.85*len(undesirable))])

test_expensive_desirable = TestDescriptions(expensive_test, "expensive", truncate=True, gpt2_type="gpt2")   
test_cheap_desirable = TestDescriptions(expensive_test, "cheap", truncate=True, gpt2_type="gpt2") 

test_expensive_undesirable = TestDescriptions(cheap_test, "expensive", truncate=True, gpt2_type="gpt2")   
test_cheap_undeisrable = TestDescriptions(cheap_test, "cheap", truncate=True, gpt2_type="gpt2")


import torch.nn.functional as F
from tqdm import trange
import math
from scipy.special import expit

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate(model, tokenizer, input, expensive=True, discriminative=False, direction=False, top_p=0.8, top_k=3, repetition=10, max_length=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device="cpu"
    print(device)
    model = model.to(device)
    model.eval()
    generated_num = 0
    return_text = ""
    best_score = 0

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(repetition):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(input)).to(device).unsqueeze(0)
            generated_without = torch.empty((1,1)).to(device)
            cumulated_probability = 0
            for i in range(max_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :]

                probability_distribution = None

                #If the direction flag is set, then instead of getting the first p words based on the cumulative sum,
                #We just take the first 10 words. This is because the scoring process takes too long.
                if (direction):
                  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                  sorted_indices_to_remove = cumulative_probs > top_p
                  sorted_indices_to_remove[..., :top_k] = 0
                  sorted_indices_to_remove[..., top_k:] = 1
                  indices_to_remove = sorted_indices[sorted_indices_to_remove]
                  logits[:, indices_to_remove] = filter_value


                  probability_distribution = F.softmax(logits, dim=-1)
                else:

                  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                  sorted_indices_to_remove = cumulative_probs > top_p
                  #sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                  sorted_indices_to_remove[..., 0] = 0
                  indices_to_remove = sorted_indices[sorted_indices_to_remove]
                  logits[:, indices_to_remove] = filter_value


                  probability_distribution = F.softmax(logits, dim=-1)

                #print(probability_distribution)

                

                #The temp_generated sentence is current sentence + next potential word
                #The scorer calculates the score for (current sentence + next potential word)
                #And then adjusts the probability of the 'next potential word'
                if (direction):
                  if (i > 10):
                    for k, x  in enumerate(probability_distribution):
                      for j, y in enumerate(x):
                        if y != 0:
                          temp_generated = torch.cat((generated_without, torch.tensor([[j]]).to(device)), dim=1)
                          temp_generated = temp_generated.to(device)
                          try:
                              weight = expit(labmt_avg_scorer.score_sentence(tokenizer.decode(list(temp_generated.squeeze().cpu().numpy()))))
                          except:
                              weight = 0.0001
                          probability_distribution[k][j] *= weight

                next_token = torch.multinomial(probability_distribution, num_samples=1)
                cumulated_probability += -math.log(torch.select(probability_distribution, 1, next_token.data[0].item()).item())
  
                generated = torch.cat((generated, next_token), dim=1)
                generated = generated.to(device)
                generated_without = torch.cat((generated_without, next_token), dim=1)
                generated_without = generated_without.to(device)
                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().cpu().numpy())
                    output_text = tokenizer.decode(output_list)
                    output_text = output_text[len(input):]
                    current_score = 0
                    
                    #If the discriminative flag is added, then add the roberta score to the total probability
                    if (discriminative):
                      roberta_score = F.softmax(torch.Tensor(roberta.predict([output_text])[1]), dim=-1)[0][1].item()
                      if not expensive:
                        roberta_score = 1 - roberta_score
                      current_score = roberta_score + cumulated_probability/len(output_text)
                    else:
                      current_score = cumulated_probability/len(output_text) 
                    if(current_score > best_score):
                      return_text = output_text
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().cpu().numpy())
              if len(output_list) == 0:
                return ''
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 

              output_text = output_text[len(input):]
              if (discriminative):
                roberta_score = F.softmax(torch.Tensor(roberta.predict([output_text])[1]), dim=-1)[0][1].item()
                if not expensive:
                  roberta_score = 1 - roberta_score
                current_score = roberta_score + cumulated_probability/len(output_text)
              else:
                current_score = cumulated_probability/len(output_text) 
              if(current_score > best_score):
                return_text = output_text
                
    return return_text



def compute_rouge(sentence, testset, number):
  hypotheses = [sentence]
  rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, skip_gap=4)

  references = [testset.get_original(number)]



  scores = rouge.evaluate_tokenized(hypotheses, references)
  return [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]


def faithfulness(sentence, testset, number):
  
  generated_lexicon = set()
  original_lexicon = set()

  tags = nltk.pos_tag(sentence.split(' '))
  for tag in tags:
    if (tag[1][:2] == 'NN' or tag[1][:2] == 'FW'):
      generated_lexicon.add(tag[0])
  
  tags = nltk.pos_tag(testset.get_original(number).split(' '))
  for tag in tags:
    if (tag[1][:2] == 'NN' or tag[1][:2] == 'FW'):
      original_lexicon.add(tag[0])
  
  return 1- (len(generated_lexicon - original_lexicon)/(len(generated_lexicon)+1))

rouge11 = 0
rouge21 = 0
rougel1 = 0
rouge12 = 0
rouge22 = 0
rougel2 = 0
rouge13 = 0
rouge23 = 0
rougel3 = 0
rouge14 = 0
rouge24 = 0
rougel4 = 0
rouge15 = 0
rouge25 = 0
rougel5 = 0
rouge16 = 0
rouge26 = 0
rougel6 = 0

faith1 = 0
faith2 = 0
faith3 = 0
faith4 = 0
faith5 = 0
faith6 = 0


for i in range(len(test_expensive_desirable)):
        sentence = generate(model, tokenizer, test_expensive_desirable[i], repetition=5, max_length=300, discriminative=False, direction=False)
        rouges = compute_rouge(sentence, test_expensive_desirable, i)
        rouge11 += rouges[0]
        rouge21 += rouges[1]
        rougel1 += rouges[2]
        faith1 += faithfulness(sentence, test_expensive_desirable, i)

        sentence = generate(model, tokenizer, test_expensive_desirable[i], repetition=5, max_length=300, discriminative=True, direction=False)
        rouges = compute_rouge(sentence, test_expensive_desirable, i)
        rouge12 += rouges[0]
        rouge22 += rouges[1]
        rougel2 += rouges[2]
        faith2 += faithfulness(sentence, test_expensive_desirable, i)

        sentence = generate(model, tokenizer, test_expensive_desirable[i], repetition=5, max_length=300, discriminative=False, direction=True)
        rouges = compute_rouge(sentence, test_expensive_desirable, i)
        rouge13 += rouges[0]
        rouge23 += rouges[1]
        rougel3 += rouges[2]
        faith3 += faithfulness(sentence, test_expensive_desirable, i)

        sentence = generate(model, tokenizer, test_expensive_desirable[i], repetition=5, max_length=300, discriminative=True, direction=True)
        rouges = compute_rouge(sentence, test_expensive_desirable, i)
        rouge14 += rouges[0]
        rouge24 += rouges[1]
        rougel4 += rouges[2]
        faith4 += faithfulness(sentence, test_expensive_desirable, i)

for i in range(len(test_expensive_desirable)):
        sentence = generate(model, tokenizer, test_cheap_desirable[i], repetition=5, max_length=300, discriminative=False, direction=False)
        rouges = compute_rouge(sentence, test_cheap_desirable, i)
        rouge15 += rouges[0]
        rouge25 += rouges[1]
        rougel5 += rouges[2]
        faith5 += faithfulness(sentence, test_cheap_desirable, i)

        sentence = generate(model, tokenizer, test_cheap_desirable[i], repetition=5, max_length=300, discriminative=True, direction=False)
        rouges = compute_rouge(sentence, test_cheap_desirable, i)
        rouge16 += rouges[0]
        rouge26 += rouges[1]
        rougel6 += rouges[2]
        faith6 += faithfulness(sentence, test_cheap_desirable, i)


print('---------------------RESULTS---------------------')
print("case 1")
print(rouge11/len(test_expensive_desirable))
print(rouge21/len(test_expensive_desirable))
print(rougel1/len(test_expensive_desirable))
print(faith1/len(test_expensive_desirable))
print()
print("case 2")
print(rouge12/len(test_expensive_desirable))
print(rouge22/len(test_expensive_desirable))
print(rougel2/len(test_expensive_desirable))
print(faith2/len(test_expensive_desirable))
print()
print("case 3")
print(rouge13/len(test_expensive_desirable))
print(rouge23/len(test_expensive_desirable))
print(rougel3/len(test_expensive_desirable))
print(faith3/len(test_expensive_desirable))
print()
print("case 4")
print(rouge14/len(test_expensive_desirable))
print(rouge24/len(test_expensive_desirable))
print(rougel4/len(test_expensive_desirable))
print(faith4/len(test_expensive_desirable))
print()
print("case 5")
print(rouge15/len(test_expensive_desirable))
print(rouge25/len(test_expensive_desirable))
print(rougel5/len(test_expensive_desirable))
print(faith5/len(test_expensive_desirable))
print()
print("case 6")
print(rouge16/len(test_expensive_desirable))
print(rouge26/len(test_expensive_desirable))
print(rougel6/len(test_expensive_desirable))
print(faith6/len(test_expensive_desirable))
print()
