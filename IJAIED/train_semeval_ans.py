import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import argparse
#import matplotlib.pyplot as plt
#import seaborn
from model_zoo.model import CMAN, ScheduledOptim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

def get_metrics_binary(test_prob,test_tensor):
    test_array = test_tensor.cpu().numpy()
    pred_prob = torch.softmax(torch.cat(test_prob),dim=1)
    _,pred = torch.max(pred_prob,1)
    pred_prob = pred_prob.cpu().detach().numpy()
    new_pred = pred.cpu().numpy()
    precision = precision_score(test_array, new_pred)
    recall = recall_score(test_array,new_pred)
    f1 = f1_score(test_array, new_pred)
    accuracy = accuracy_score(test_array, new_pred)
    auc = roc_auc_score(test_array,pred_prob[:,1])
    print('acc:{}, precision:{}, recall:{}, f1:{}, AUC:{} '.format(accuracy,precision,recall,f1, auc))
    return accuracy,precision,recall,f1, auc

def cal_performence(pred, gold, class_weight=None, smoothing=False):
    '''apply label smooth if needed'''
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, gold)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.eq(gold).sum().item()
    return loss, n_correct

def train_epoch(model, train_data, optimizer, device):
    '''Epoch operation in training phase'''
    model.train()
    total_loss = 0
    n_total = 0
    total_correct_label=0
    
    for batch in tqdm(train_data, mininterval=2, desc = ' -(Training) ', leave=False):
        left_train, right_train, train_label = map(lambda x: x.to(device), batch)
        
        optimizer.zero_grad()
        _,_,_,_, pred = model(left_train, right_train)
        
        loss, n_correct_label = cal_performence(pred, train_label)
        
        loss.backward()
        
        # optimizer.step_and_update_lr()
        optimizer.step()
        
        # note keeping
        total_loss += loss.item()
        total_correct_label += n_correct_label
        n_total += len(left_train)
    
    accuracy_label = total_correct_label/n_total
    loss_per_sample = total_loss / n_total
    return accuracy_label, loss_per_sample

def eval_epoch(model, validation_data, device):
    '''Epoch operation in validation phase'''
    model.eval()
    total_loss = 0
    n_total = 0
    total_correct_label=0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=' -(Validation) ',leave = False):
            left_dev, right_dev, dev_label = map(lambda x: x.to(device), batch)

            _,_,_,_, pred = model(left_dev, right_dev)

            loss, n_correct_label = cal_performence(pred, dev_label)

            # note keeping
            total_loss += loss.item()
            total_correct_label += n_correct_label
            n_total += len(left_dev)
    
    accuracy_label = total_correct_label/n_total
    loss_per_sample = total_loss / n_total
    return accuracy_label, loss_per_sample


def test_epoch(model, test_data, device):
    '''Epoch operation in validation phase'''
    model.eval()
    total_loss = 0
    n_total = 0
    total_correct_label=0
    test_predict_prob = []
    test_pred_label = []
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2, desc=' -(Test) ',leave = False):
            left_test, right_test, test_label = map(lambda x: x.to(device), batch)

            _,_,_,_, pred = model(left_test, right_test)

            loss, n_correct_label = cal_performence(pred, test_label)

            # note keeping
            total_loss += loss.item()
            total_correct_label += n_correct_label
            n_total += len(left_test)
            test_predict_prob.append(pred.cpu())
            test_pred_label.append(test_label.cpu())
    test_pred_label = torch.cat(test_pred_label)

    get_metrics_binary(test_predict_prob,test_pred_label)
    accuracy_label = total_correct_label/n_total
    loss_per_sample = total_loss / n_total
    return accuracy_label, loss_per_sample


def train(model, training_data, validation_data, test_data_ans, optimizer, device, opt):
    '''Start training'''
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performence will be written to file: {} and {}'.format(log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file,'w') as log_vf:
            log_tf.write('epoch,loss_per_sample, accuracy_label\n')
            log_vf.write('epoch,loss_per_sample, accuracy_label\n')
    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch',epoch_i,' ]')
        start = time.time()
        train_label_accu, train_loss = train_epoch(
            model, training_data, optimizer, device
        )
        print(' -(Trianing) train_loss: {t_loss: 8.5f}, label_acc: {label_acc: 3.3f}, elapse: {elapse:3.3f} min'.format(
            t_loss=train_loss,label_acc=train_label_accu, elapse=(time.time()-start)/60))
        start = time.time()
        valid_accu, valid_loss = eval_epoch(model,validation_data, device)
        print(' -(Validation) dev_loss: {t_loss: 8.5f}, dev_label_acc: {label_acc: 3.3f}, elapse: {elapse:3.3f} min'.format(
            t_loss=valid_loss,label_acc=valid_accu, elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }
        if valid_accu >= max(valid_accus):
                test_accu, test_loss = test_epoch(model,test_data_ans,device)
                print('[Info]: For question task New Test accuracy {}'.format(test_accu))
        
        if log_train_file and log_valid_file:
            with open(log_train_file,'a') as log_tf, open(log_valid_file,'a') as log_vf:
                log_tf.write('{epoch}, {loss: 8.5f},{accu_label: 3.3f}\n'.format(
                    epoch = epoch_i, loss=train_loss, accu_label=100*train_label_accu
                ))
                log_vf.write('{epoch}, {loss: 8.5f},{accu_label: 3.3f}\n'.format(
                    epoch = epoch_i, loss=valid_loss, accu_label=100*valid_accu
                ))

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            if opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print(' -[Info] The check point file has been updated.')


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size',type=int, default=32)
    
    parser.add_argument('-d_model',type=int, default=128)
    parser.add_argument('-number_block',type=int, default=3)
    parser.add_argument('-head_number',type=int, default=8)
    parser.add_argument('-d_ff',type=int, default=512)
    parser.add_argument('-class_num',type=int, default=2)
    parser.add_argument('-seq_len',type=int, default=30)

    parser.add_argument('-n_warmup_steps',type=int, default=2000)
    
    parser.add_argument('-dropout',type=float,default=0.3)
    parser.add_argument('-emb_dropout',type=float,default=0.1)
    
    parser.add_argument('-log',type=str, default='./logs/2way_sem_ans')
    parser.add_argument('-save_model',type=str, default='./saved_model/2way_sem_ans')
    parser.add_argument('-save_mode', type=str, choices=['all','best'], default='best')

    opt = parser.parse_args()
    print(opt)
    
    batch_size=opt.batch_size

    # load preprocessed data
    # answer for test data
    test_left_answer = torch.LongTensor(torch.load("./semeval/processed_data/2way/unseen_answers/left_test_answer"))
    test_right_answer = torch.LongTensor(torch.load("./semeval/processed_data/2way/unseen_answers/right_test_answer"))

    test_left_question = torch.LongTensor(torch.load("./semeval/processed_data/2way/unseen_questions/left_test_question"))
    test_right_question = torch.LongTensor(torch.load("./semeval/processed_data/2way/unseen_questions/right_test_question"))

    test_left_dom = torch.LongTensor(torch.load("./semeval/processed_data/2way/unseen_domains/left_test_domain"))
    test_right_dom = torch.LongTensor(torch.load("./semeval/processed_data/2way/unseen_domains/right_test_domain"))
    #yaoyongde train data
    train_data_left = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/2way/left_train'))
    train_data_right = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/2way/right_train'))
    #yaoyongde dev data
    dev_data_left = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/2way/left_dev'))
    dev_data_right = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/2way/right_dev'))

    #yaoyongde label
    train_label = torch.FloatTensor(torch.load('./semeval/sem_new_train_dev/2way/train_label')).long()
    test_label_answer = torch.FloatTensor(torch.load('./semeval/processed_data/2way/unseen_answers/test_label_answer')).long()
    dev_label = torch.FloatTensor(torch.load('./semeval/sem_new_train_dev/2way/dev_label')).long()
    test_label_domain = torch.FloatTensor(torch.load('./semeval/processed_data/2way/unseen_domains/test_label_domain')).long()
    test_label_question = torch.FloatTensor(torch.load('./semeval/processed_data/2way/unseen_questions/test_label_question')).long()

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_data_left, train_data_right, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(dev_data_left, dev_data_right, dev_label)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our answer test set.
    test_data_ans = TensorDataset(test_left_answer, test_right_answer, test_label_answer)
    test_sampler_ans = SequentialSampler(test_data_ans)
    test_dataloader_ans = DataLoader(test_data_ans, sampler=test_sampler_ans, batch_size=batch_size)


    vocabulary = torch.load('./semeval/processed_data/2way/word_vocab')
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.vocab_size = len(vocabulary)

    model = CMAN(
        d_model=opt.d_model, 
        number_block=opt.number_block, 
        head_number=opt.head_number,
        d_ff=opt.d_ff, 
        class_num=opt.class_num,
        seq_len=opt.seq_len,
        vocab_size=opt.vocab_size,
        drop_out=opt.dropout,
        emb_dropout = opt.emb_dropout).to(opt.device)

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         filter(lambda x: x.requires_grad, model.parameters()),
    #         betas=(0.9,0.98),eps=1e-09),
    #     opt.d_model, opt.n_warmup_steps
    # )
    optimizer = optim.Adadelta(model.parameters(),lr=0.1,rho=0.9,eps=1e-6,weight_decay=0)
    train(model, train_dataloader, validation_dataloader, test_dataloader_ans, optimizer, opt.device, opt)

if __name__ == "__main__":
    main()





