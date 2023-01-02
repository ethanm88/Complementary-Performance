import preprocess_stance as preprocess_stance
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn
import torch
from  torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import csv
MAX_LENGTH = 512

class SaliencyModel(nn.Module):
  def __init__(self, encoder, num_labels, regressor_in=768, regressor_out=1):
    super().__init__()
    self.num_labels = num_labels
    self.encoder = encoder
    self.regressor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(regressor_in, regressor_out),
            nn.Sigmoid()
    )

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    output = self.encoder(input_ids)
    cls_token = output.last_hidden_state[:,0,:]
    logits = self.regressor(cls_token)[:,0]
    if labels is not None:
      avg_loss = F.mse_loss(logits, labels.float())

      return avg_loss, logits
    else:
      return logits

def eval(smodel, tokenizer, test_dataset, bs = 1):
    smodel.eval()
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open('local_search.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["initial_text", "initial_difference", "final_text", "final_difference", "pred_y", "true_y"])
    for batch_idx, batch in enumerate(test_loader):
        print("Tweet Number:", batch_idx)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels)
        difference = abs(1 - outputs[1].detach().item())

        initial_text = batch['texts'][0]
        pred_y = batch['pred_y'][0]
        true_y = batch['Y'][0]
        current_text = initial_text
        print('Initial Text:', initial_text)

        initial_difference = difference
        print('Initial Difference:', initial_difference)
        try:
          while True:
              neighbor_dataset, neighbor_text = preprocess_stance.get_local_neighbors(current_text, batch['labels'], tokenizer)
              neighbor_dataloader = DataLoader(neighbor_dataset, batch_size=bs)
              found_new = False
              for inner_batch_idx, inner_batch in enumerate(neighbor_dataloader):
                  input_ids = inner_batch['input_ids'].to(device)
                  attention_mask = inner_batch['attention_mask'].to(device)
                  labels = inner_batch['labels'].to(device)
                  outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels)
                  current_difference = abs(1 - outputs[1].detach().item())
                  if current_difference < difference:
                      difference = current_difference
                      current_text = neighbor_text[inner_batch_idx]
                      found_new = True
              if not found_new:
                break
              print('Current Difference:', difference)
          print('Final Difference:', difference)
          final_text = current_text
          final_difference = difference
          print('Final Text:', final_text)
          print()
          with open('local_search.csv', 'a') as f:
              writer = csv.writer(f)
              writer.writerow([initial_text, initial_difference, final_text, final_difference, pred_y, true_y])
        except Exception as e:
          print(e)
          continue


def train(lr=2e-7, bs = 8, num_epochs = 3, warmup_ratio = 0.1, from_save=False):
    preprocess_stance.set_seed(30)
    saliency_markers = ['<sal0>','</sal0>','<sal1>','</sal1>']
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(saliency_markers)
    train_dataset, test_dataset, val_dataset, mean_score = preprocess_stance.create_data_instances(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    if from_save:
        smodel = torch.load("/srv/share5/emendes3/saliency_model.pt")
        smodel.cuda()
        return smodel, tokenizer, test_dataset, None, None

    encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    encoder.resize_token_embeddings(len(tokenizer))
    smodel = SaliencyModel(encoder, MAX_LENGTH)
    smodel.cuda()
    smodel.train()

    num_train_steps = int(len(train_dataset) / bs * num_epochs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    param_optimizer = list(smodel.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optim = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=int(0.1* num_train_steps),
                                                num_training_steps=num_train_steps)
    '''
    Training
    '''
    AVG_TRAIN_SCORE_BASELINE = mean_score
    best_mse = 999999999
    best_epoch = 0 # keep track of best epoch
    current_mse = 0
    best_smodel = None
    for epoch in range(num_epochs):
        print("Epoch: ", epoch, end="  ")
        smodel.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            if batch_idx + 1 == len(train_loader):
                optim.step()
                scheduler.step()
                optim.zero_grad()
        avg_train_loss = total_loss / len(train_loader)
        print("Average train loss: {}".format(avg_train_loss))
        
        '''
        Evaluation
        '''
        smodel.eval()
        y_hat = []
        y_true = []
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels)
            outputs = outputs[1].detach().cpu()
            y_hat += outputs.tolist()
            y_true += labels.detach().tolist()
        print('MSE on Validation:', F.mse_loss(torch.tensor(y_hat), torch.tensor(y_true)).item())
        current_mse = F.mse_loss(torch.tensor(y_hat), torch.tensor(y_true)).item()

        avg_mse = F.mse_loss(torch.tensor(y_true),  AVG_TRAIN_SCORE_BASELINE * torch.ones(len(y_hat))).item()
        one_mse = F.mse_loss(torch.tensor(y_true), 1.0 * torch.ones(len(y_hat))).item()
        print('Avg MSE on Validation:', avg_mse)
        print('One MSE on Validation:', one_mse)
        if best_mse > current_mse:
          best_mse = current_mse
          best_epoch = epoch
          best_smodel = smodel    
        smodel.train()
    torch.save(best_smodel, "/srv/share5/emendes3/saliency_model.pt")
    return smodel, tokenizer, test_dataset, best_mse, best_epoch

def hypertrain():
  MAX_EPOCHS_TESTED = 20
  #2e-6, 3e-6, 5e-6, 8e-6, 1e-5, 4e-5, 7e-5, 9e-5
  # generate hyper params to test 
  params = []
  for lr in [9e-7, 2e-6, 3e-6, 5e-6, 8e-6, 1e-5, 4e-5, 7e-5, 9e-5]:
    for batch_size in [4, 8, 16, 32]:
      params.append({
        'lr': lr,
        'batch_size': batch_size
      })
  
  # brute force search for best hyper params
  min_mse = 999999999
  ideal_params = None
  for idx, p in enumerate(params):
    print('Testing:', idx + 1, "of", len(params), flush=True)
    _, _, _, mse, best_epoch = train(bs = p["batch_size"], lr=p["lr"], num_epochs = MAX_EPOCHS_TESTED)
    p['epoch'] = best_epoch + 1 
    if mse < min_mse:
      ideal_params = p
      min_mse = mse
      print('New Best Parameters:', ideal_params, end=' ', flush=True)
      print('Best MSE:', min_mse, flush=True)
  print(flush=True)
  print("Ideal", ideal_params, min_mse, flush=True)
if __name__ == "__main__":
  #hypertrain()
  smodel, tokenizer, test_dataset, mse, _ = train(bs = 32, lr=9e-05, num_epochs=5)
  eval(smodel, tokenizer, test_dataset)

#Ideal {'lr': 9e-05, 'batch_size': 32, 'epoch': 14} 0.00567991565912962