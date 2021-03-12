# all methods to train, test, predict
import os
import tqdm
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import multiprocessing
import random
import pickle
from model import torch_utils
from model.data_generator import AutoCompleteDataset
from model.gru_model import AutoCompleteNet


DATA_PATH = os.path.join(Path(__file__).parent, '..', 'data')

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def perplexity_loss(loss):
  if type(loss)!='torch.Tensor':
    loss = torch.tensor(loss)
  return torch.exp(loss).tolist()

def predict_next_characters(automodel, device, seed_words, vocab=None, num_chars=3):
    """
    using given model it predicts best n chars
    """
    if not vocab:
        with open(f'{DATA_PATH}/vocabulary.pkl','rb') as f:
            vocab = pickle.load(f)
    automodel.eval()
    with torch.no_grad():
        arr = torch.LongTensor([vocab['voc2ind'].get(char,1) for char in seed_words])

        # Computes the initial hidden state from the prompt (seed words).
        hidden = None
        output=None
        for ind in arr:
            data = ind.to(device)
            output, hidden = automodel.inference(data, hidden)
        # get top n probs indexes
        # get top 3 values's indices
        ind_arr = torch.topk(output,num_chars)[1].tolist()
        char_arr = [vocab['ind2voc'].get(ind,'_UNK_') for ind in ind_arr[0]]
    return char_arr


def train_one_epoch(automodel, device, optimizer, train_loader, lr, epoch, log_interval):
    automodel.train()
    losses = []
    pp_H = []
    hidden = None
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader,miniters=10)):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches. 
        # Otherwise the backward would try to go all the way to the beginning every time.
        if hidden is not None:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = automodel(data)
        pred = output.max(-1)[1]
        loss = automodel.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pp_H = perplexity_loss(loss)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tPerplexity: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), pp_H))
    return np.mean(losses), perplexity_loss(np.mean(losses))


def eval(automodel, device, eval_loader,log_interval):
    automodel.eval()
    eval_loss = 0
    correct = 0
    iter = enumerate(eval_loader)
    with torch.no_grad():
        hidden = None
        for batch_idx,(data , label) in enumerate(eval_loader):
            data, label = data.to(device), label.to(device)
            try:
                output, hidden = automodel(data, hidden)
                eval_loss += automodel.loss(output, label, reduction='mean').item()
                pred = output.max(-1)[1]
                correct_mask = pred.eq(label.view_as(pred))
                num_correct = correct_mask.sum().item()
                correct += num_correct
                # Comment this out to avoid printing test results
                if batch_idx % log_interval == 0:
                    print(f'Input\t{eval_loader.dataset.array_to_sentence(data[0])}\n \
                    GT\t{eval_loader.dataset.array_to_sentence(label[0])}\n \
                    pred\t{eval_loader.dataset.array_to_sentence(pred[0])}\n\n')
            except:
                print(data, label)
                raise

    eval_loss /= len(eval_loader)
    eval_accuracy = 100. * correct / (len(eval_loader.dataset) * eval_loader.dataset.sequence_length)
    pp_H = perplexity_loss(eval_loss)
    print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), perplexity {}\n'.format(
        eval_loss, correct, len(eval_loader.dataset) * eval_loader.dataset.sequence_length,
        100. * correct / (len(eval_loader.dataset) * eval_loader.dataset.sequence_length),pp_H))
    return eval_loss, eval_accuracy, pp_H

def test(automodel,device):
    automodel.eval()
    eval_loss = 0
    correct = 0
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
            'pin_memory': True} if torch.cuda.is_available() else {}
    data_eval = AutoCompleteDataset(data_file=f'{DATA_PATH}/masterdata/master_test.txt.pkl',sequence_length=100,batch_size=128)
    eval_loader = torch.utils.data.DataLoader(data_eval, batch_size=128,
                                            shuffle=False, **kwargs)
    iter = enumerate(eval_loader)
    with torch.no_grad():
        hidden = None
        for batch_idx,(data , label) in enumerate(eval_loader):
            data, label = data.to(device), label.to(device)
            try:
                output, hidden = automodel(data, hidden)
                eval_loss += automodel.loss(output, label, reduction='mean').item()
                pred = output.max(-1)[1]
                correct_mask = pred.eq(label.view_as(pred))
                num_correct = correct_mask.sum().item()
                correct += num_correct
                # Comment this out to avoid printing test results
                if batch_idx % 5000 == 0:
                    print(f'Input\t{eval_loader.dataset.array_to_sentence(data[0])}\n \
                    GT\t{eval_loader.dataset.array_to_sentence(label[0])}\n \
                    pred\t{eval_loader.dataset.array_to_sentence(pred[0])}\n\n')
            except:
                print(data, label)
                raise

    eval_loss /= len(eval_loader)
    eval_accuracy = 100. * correct / (len(eval_loader.dataset) * eval_loader.dataset.sequence_length)
    pp_H = perplexity_loss(eval_loss)
    print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), perplexity {}\n'.format(
        eval_loss, correct, len(eval_loader.dataset) * eval_loader.dataset.sequence_length,
        100. * correct / (len(eval_loader.dataset) * eval_loader.dataset.sequence_length),pp_H))
    return eval_loss, eval_accuracy, pp_H

def train_eval(work_dir='./logs'):
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 512
    FEATURE_SIZE = 256
    EVAL_BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.000005
    USE_CUDA = True
    PRINT_INTERVAL = 10000
    LOG_PATH = f'{work_dir}/log.pkl'
    checkpoints = f'{work_dir}/checkpoint'
    
    print('Creating Training data set')
    data_train = AutoCompleteDataset(data_file=f'{DATA_PATH}/masterdata/master_train.txt.pkl',sequence_length=SEQUENCE_LENGTH,batch_size=BATCH_SIZE)
    print(f"data_train vocab size {data_train.vocab_size()}")
    print('Creating eval data set')
    data_eval = AutoCompleteDataset(data_file=f'{DATA_PATH}/masterdata/master_dev.txt.pkl',sequence_length=SEQUENCE_LENGTH,batch_size=EVAL_BATCH_SIZE)

    vocab = data_train.vocab

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
            'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                            shuffle=True, **kwargs)
    eval_loader = torch.utils.data.DataLoader(data_eval, batch_size=EVAL_BATCH_SIZE,
                                            shuffle=False, **kwargs)

    automodel = AutoCompleteNet(data_train.vocab_size(), FEATURE_SIZE).to(device)

    optimizer = optim.Adam(automodel.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = automodel.load_last_model(checkpoints)

    train_losses, eval_losses, eval_accuracies , eval_perplexities, train_perplexities = torch_utils.read_log(LOG_PATH, ([], [], [],[],[]))
    eval_loss, eval_accuracy, eval_perplexity = eval(automodel, device, eval_loader,PRINT_INTERVAL)

    eval_losses.append((start_epoch, eval_loss))
    eval_accuracies.append((start_epoch, eval_accuracy))
    eval_perplexities.append((start_epoch, eval_perplexity))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            if epoch % 10 == 0 :
                LEARNING_RATE = LEARNING_RATE *.8
            optimizer = optim.Adam(automodel.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            train_loss, train_perplexity = train_one_epoch(automodel, device, optimizer, train_loader, LEARNING_RATE, epoch, PRINT_INTERVAL)
            test_loss, test_accuracy, test_perplexity = eval(automodel, device, eval_loader,PRINT_INTERVAL)
            train_losses.append((epoch, train_loss))
            eval_losses.append((epoch, test_loss))
            eval_accuracies.append((epoch, test_accuracy))
            eval_perplexities.append((epoch, test_perplexity))
            train_perplexities.append((epoch, train_perplexity))
            torch_utils.write_log(LOG_PATH, (train_losses, eval_losses, eval_accuracies, eval_perplexities, train_perplexities))
            automodel.save_best_model(test_accuracy, checkpoints + '/%03d.pt.checkpoint' % epoch)
            seed_words = 'Do you know '
    
            for ii in range(10):
                next_chars = predict_next_characters(automodel, device, seed_words, vocab, num_chars=3)
                seed_words = seed_words + random.choice(next_chars)
                print(f'generated sample \t{next_chars} \t {seed_words}')
            print('')

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('Saving final model')
        automodel.save_model(checkpoints+'/%03d.pt' % epoch, 0)
        ep, val = zip(*train_losses)
        torch_utils.plot(ep, val, 'Train loss', 'Epoch', 'Error')
        ep, val = zip(*eval_losses)
        torch_utils.plot(ep, val, 'Eval loss', 'Epoch', 'Error')
        ep, val = zip(*eval_accuracies)
        torch_utils.plot(ep, val, 'Eval accuracy', 'Epoch', 'Error')
        ep, val = zip(*eval_perplexities)
        torch_utils.plot(ep, val, 'Eval perplexity', 'Epoch', 'Error')
        ep, val = zip(*train_perplexities)
        torch_utils.plot(ep, val, 'Train perplexity', 'Epoch', 'Error')
        return automodel, vocab, device

if __name__=='__main__':
    automodel, vocab, device = train_eval()
