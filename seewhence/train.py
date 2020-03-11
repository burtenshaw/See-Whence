import time
import torch
from torchtext import data

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def f1_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    return f1_score(rounded_preds, y, average='macro')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_iterators(train_data, test_data, BATCH_SIZE = 128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data), 
        batch_size = BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch = True,
        device = device)
    return train_iterator, test_iterator

def predict(model, iterator):
    model.eval()
    y=[]
    with torch.no_grad():
        for batch in iterator:
            preds = [round(x[0],2) for x in torch.sigmoid(model(batch)).tolist()]
            labels = batch.label.tolist()
            indexes = batch.index
            y.extend(list(zip(indexes,
                              labels,
                              preds)))
    return y