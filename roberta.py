import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import pickle
import wandb

def create_dataset(desirable, undesirable):
  df_to_return = pd.DataFrame(columns = ["text", "labels"])
  
  desirable['labels'] = 1
  desirable.rename(columns = {'description': "text"}, inplace = True)

  undesirable['labels'] = 0
  undesirable.rename(columns = {'description':'text'}, inplace = True)

  frames = [df_to_return, desirable[['text', 'labels']], undesirable[['text', 'labels']]]
  df_to_return = pd.concat(frames, ignore_index=True)
  return df_to_return



desirable = pd.read_csv('expensive.csv')
undesirable = pd.read_csv('cheap.csv')

_, validate, _ = \
              np.split(desirable.sample(frac=1, random_state=1), 
                       [int(.7*len(desirable)), int(.85*len(desirable))])

_, validate2, _ = \
              np.split(undesirable.sample(frac=1, random_state=1), 
                       [int(.7*len(undesirable)), int(.85*len(undesirable))])

roberta_df = create_dataset(validate, validate2)

traindata, validate, test = np.split(roberta_df.sample(frac=1, random_state=1), 
                       [int(.7*len(roberta_df)), int(.85*len(roberta_df))])
                                           
traindata.to_csv('roberta-train.csv', sep ='\t', index=False, header=False)
validate.to_csv('roberta-validate.csv', sep ='\t', index=False, header=False)
test.to_csv('roberta-test.csv', sep ='\t', index=False, header=False)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

sweep_config = {
        "method": "bayes",
        "metric": {"name": "mcc", "goal":"maximize"},
        "parameters": {
                "num_train_epochs": {"values":[2,3,4]},
                "learning_rate": {"min":5e-5, "max": 4e-4},
        },
}

#sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")
model_args = ClassificationArgs(learning_rate=2e-5, num_train_epochs=4,use_multiprocessing=True, train_batch_size=16, eval_batch_size=8, overwrite_output_dir=True, evaluate_during_training=True)
model_args.lazy_loading = True
# Create a TransformerModel

def train():
#       wandb.init()
        model = ClassificationModel('roberta', 'roberta-base', use_cuda=True, args=model_args)

        # Train the model
        model.train_model('roberta-train.csv', eval_df='roberta-validate.csv')

        # Evaluate the model
        result, _ ,_ = model.eval_model('roberta-test.csv')
        print(result)
        pickle.dump(model, open('roberta.pkl', 'wb'))

train()
#wandb.agent(sweep_id, train)
