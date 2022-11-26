import preprocess_covid as preprocess_covid
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn
import torch
from  torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import csv
MAX_LENGTH = 128

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
  def __init__(self, encoder, num_labels):
    super().__init__()
    self.num_labels = num_labels
    self.encoder = encoder
    self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
    self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
    #self.encoder.init_weights()

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e_span=None):
    output = self.encoder(input_ids)
    sequence_output = output[0]
    batchsize, _, _ = sequence_output.size()
    batch_index = [i for i in range(batchsize)]
    repr = sequence_output[batch_index, e_span]

    cls_token = self.dropout(repr)
    logits = self.classifier(cls_token).reshape(batchsize)
    if labels is not None:
      avg_loss = F.binary_cross_entropy(torch.sigmoid(logits), labels.float())

      return avg_loss, torch.sigmoid(logits)
    else:
      return torch.sigmoid(logits)

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
        e_span = batch['e_span'].to(device)

        outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels, e_span=e_span)
        difference = abs(1 - outputs[1].detach().item())

        initial_text = test_text[batch_idx]
        current_text = initial_text
        print('Initial Text:', initial_text)

        initial_difference = difference
        print('Initial Difference:', initial_difference)
        try:
          while True:
              neighbor_dataset, neighbor_text = preprocess_covid.get_local_neighbors(current_text, batch['labels'], tokenizer)
              neighbor_dataloader = DataLoader(neighbor_dataset, batch_size=bs)
              found_new = False
              for inner_batch_idx, inner_batch in enumerate(neighbor_dataloader):
                  input_ids = inner_batch['input_ids'].to(device)
                  attention_mask = inner_batch['attention_mask'].to(device)
                  labels = inner_batch['labels'].to(device)
                  e_span = inner_batch['e_span'].to(device)
                  outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels, e_span=e_span)
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
              writer.writerow([initial_text, initial_difference, final_text, final_difference, outputs[1].detach().item()])
        except:
          continue


def train(lr=1e-6, bs = 4, num_epochs = 400, warmup_ratio = 0.1, from_save=True):
    saliency_markers = ['<in_sal>', '<out_sal>', '<cp>']
    tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
    tokenizer.add_tokens(saliency_markers)
    train_dataset, test_dataset, test_text = preprocess_covid.create_data_instances(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    if from_save:
        encoder = BertModel.from_pretrained("/srv/share5/emendes3/covid_misiformation_saliency_model", output_hidden_states=True)
        smodel = SaliencyModel(encoder, MAX_LENGTH)
        smodel.cuda()
        return smodel, tokenizer, test_dataset, test_text

    encoder = BertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', output_hidden_states=True)
    saliency_markers = ['<in_sal>', '<out_sal>', '<cp>']
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
    
    best_f1 = 0
    best_model = None
    accum_iter = 2 # gradient accumulation
    for epoch in range(num_epochs):
        print("Epoch: ", epoch, end="  ")
        smodel.train()
        count = 0
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            #print(batch)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            e_span = batch['e_span'].to(device)
            outputs = smodel(input_ids, attention_mask=attention_mask, labels=labels, e_span=e_span)

            loss = outputs[0]
            loss = loss / accum_iter # gradient accumulation

            total_loss += loss.item()
            loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)): # gradient accumulation
                optim.step()
                scheduler.step()
                optim.zero_grad()
            count += bs
        avg_train_loss = total_loss / len(train_loader)
        print("Average train loss: {}".format(avg_train_loss))
        #f1 = evaluate_cp(model, dev_dataset)
        """ if f1 > best_f1:
            best_f1 = f1
            best_model = model
        """
    smodel.encoder.save_pretrained("/srv/share5/emendes3/covid_misiformation_saliency_model")
    return smodel, tokenizer, test_dataset, test_text
if __name__ == "__main__":
    smodel, tokenizer, test_dataset, test_text = train()
    eval(smodel, tokenizer, test_dataset, test_text)