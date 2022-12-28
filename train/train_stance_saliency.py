import preprocess_stance as preprocess_stance
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn
import torch
from  torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import csv
MAX_LENGTH = 512

'''
class SaliencyModel(nn.Module):
  def __init__(self, encoder, concat_size=MAX_LENGTH):
    super().__init__()
    self.concat_size = concat_size
    self.encoder = encoder
    self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
    self.fcc_bert_1 = nn.Linear(self.encoder.config.hidden_size, self.concat_size)
    self.fcc_bert_2 = nn.Linear(self.concat_size, (int)(self.concat_size/4))
    self.fcc_bert_3 = nn.Linear((int)(self.concat_size/4), 1)
    self.fcc_saliency = nn.Linear(MAX_LENGTH, self.concat_size)
    self.fcc_concat_1 = nn.Linear(self.concat_size * 2, self.concat_size)
    self.fcc_concat_2 = nn.Linear(self.concat_size, (int)(self.concat_size / 4))
    self.fcc_concat_3 = nn.Linear((int)(self.concat_size / 4), 1)

  def forward(self, input_ids, saliency_scores, token_type_ids=None, attention_mask=None, labels=None):
    self.saliency_scores = saliency_scores
    self.saliency_scores.requires_grad = True
    output = self.encoder(input_ids, token_type_ids, attention_mask)
    sequence_output = output[0]
    batchsize, _, _ = sequence_output.size()
    batch_index = [i for i in range(batchsize)]
    repr = sequence_output[batch_index]

    cls_token = self.dropout(repr)

    bert_logits = torch.squeeze(self.fcc_bert_3(self.fcc_bert_2(self.fcc_bert_1(cls_token))), dim=2)
    saliency_logits = self.fcc_saliency(self.saliency_scores)
    concat_logits = torch.cat((bert_logits, saliency_logits), 1)
    logits = self.fcc_concat_3(self.fcc_concat_2(self.fcc_concat_1(concat_logits)))
    loss = nn.MSELoss()
    avg_loss = loss(logits.clone().reshape(-1).float(), labels.float())
    print(logits.clone().reshape(-1).float(), labels.float())
    return avg_loss, logits
'''
class SaliencyModel(nn.Module):
  def __init__(self, encoder, num_labels, regressor_in=768, regressor_out=1):
    super().__init__()
    self.num_labels = num_labels
    self.encoder = encoder
    #self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
    #self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)
    self.regressor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(regressor_in, regressor_out),
            nn.Sigmoid()
    )

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    output = self.encoder(input_ids)
    cls_token = output.last_hidden_state[:,0,:]
    logits = self.regressor(cls_token)[:,0]
    '''
    sequence_output = output[0]
    batchsize, _, _ = sequence_output.size()
    batch_index = [i for i in range(batchsize)]
    repr = sequence_output[batch_index, e_span]

    cls_token = self.dropout(repr)
    logits = self.regressor(cls_token).reshape(batchsize)
    '''
    if labels is not None:
      avg_loss = F.mse_loss(logits, labels.float())

      return avg_loss, logits
    else:
      return logits

def eval(smodel, tokenizer, test_dataset, test_text, bs = 1):
    smodel.eval()
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open('local_search.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["initial_text", "initial_difference", "final_text", "final_difference", "label"])
    for batch_idx, batch in enumerate(test_loader):
        print("Tweet Number:", batch_idx)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        #e_span = batch['e_span'].to(device)
        outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels)
        difference = abs(1 - outputs[1].detach().item())

        initial_text = test_text[batch_idx]
        current_text = initial_text
        print('Initial Text:', initial_text)

        initial_difference = difference
        print('Initial Difference:', initial_difference)
        try:
          while True:
              neighbor_dataset, neighbor_text = preprocess_stance.get_local_neighbors(current_text, batch['labels'], tokenizer)
              neighbor_dataloader = DataLoader(neighbor_dataset, batch_size=bs)
              print(len(neighbor_dataset))
              found_new = False
              for inner_batch_idx, inner_batch in enumerate(neighbor_dataloader):
                  input_ids = inner_batch['input_ids'].to(device)
                  attention_mask = inner_batch['attention_mask'].to(device)
                  labels = inner_batch['labels'].to(device)
                  outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels)
                  current_difference = abs(1 - outputs[1].detach().item())
                  print(current_difference)
                  print(neighbor_text[inner_batch_idx])
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
              writer.writerow([initial_text, initial_difference, final_text, final_difference, outputs[1].detach().item()])
        except:
          continue


def train(lr=2e-7, bs = 8, num_epochs = 3, warmup_ratio = 0.1, from_save=False):
    preprocess_stance.set_seed(30)
    saliency_markers = ['<sal0>','</sal0>','<sal1>','</sal1>','<out_sal>','</out_sal>']
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(saliency_markers)
    train_dataset, test_dataset, val_dataset, test_text = preprocess_stance.create_data_instances(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    if from_save:
        encoder = BertModel.from_pretrained("/srv/share5/emendes3/amzbook_saliency_model", output_hidden_states=True)
        smodel = SaliencyModel(encoder, MAX_LENGTH)
        smodel.cuda()
        return smodel, tokenizer, test_dataset, test_text, None, None

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
    AVG_TRAIN_SCORE_BASELINE = 0.8767871389180644
    best_mse = 999999999
    best_epoch = 0 # keep track of best epoch
    current_mse = 0
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
        smodel.train()
    smodel.encoder.save_pretrained("/srv/share5/emendes3/amzbook_saliency_model")
    return smodel, tokenizer, test_dataset, test_text, best_mse, best_epoch

def hypertrain():
  MAX_EPOCHS_TESTED = 20
  
  # generate hyper params to test 
  params = []
  for lr in [2e-6, 5e-6, 8e-6, 1e-5, 4e-5, 7e-5, 1e-4]:
    for batch_size in [4, 8, 16, 32]:
      params.append({
        'lr': lr,
        'batch_size': batch_size
      })
  
  # brute force search for best hyper params
  min_mse = 999999999
  ideal_params = None
  for idx, p in enumerate(params):
    print('Testing:', idx + 1, "of", len(params))
    _, _, _, _, mse, best_epoch = train(bs = p["batch_size"], lr=p["lr"], num_epochs = MAX_EPOCHS_TESTED)
    p['epoch'] = best_epoch 
    if mse < min_mse:
      ideal_params = p
      min_mse = mse
      print('New Best Parameters:', ideal_params, end=' ')
      print('Best MSE:', min_mse)
  print()
  print("Ideal", ideal_params, min_mse)
if __name__ == "__main__":
  #hypertrain()
  smodel, tokenizer, test_dataset, test_text, mse, _ = train(bs = 32, lr=2e-06, num_epochs=19)
  eval(smodel, tokenizer, test_dataset, test_text)


#Ideal {'lr': 2e-06, 'batch_size': 32, 'epoch': 19} 0.008832832798361778