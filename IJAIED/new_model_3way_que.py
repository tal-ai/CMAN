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
from model_zoo.new_model import CMAN_plus
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def get_metrics_weighted(test_prob,test_tensor):
    test_array = test_tensor.cpu().numpy()
    true_label_mat = label_binarize(test_array, classes=[0,1,2])
    pred_prob = torch.softmax(torch.cat(test_prob),dim=1)
    _,pred = torch.max(pred_prob,1)
    pred_prob = pred_prob.cpu().detach().numpy()
    new_pred = pred.cpu().numpy()
    
    precision = precision_score(test_array, new_pred,average='weighted')
    recall = recall_score(test_array,new_pred,average='weighted')
    f1 = f1_score(test_array, new_pred, average='weighted')
    accuracy = accuracy_score(test_array, new_pred)
    auc = roc_auc_score(true_label_mat,pred_prob,average='micro')
    print('Micro Result: acc:{}, precision:{}, recall:{}, f1:{}, AUC:{} '.format(accuracy,precision,recall,f1, auc))

def get_metrics_binary(test_prob,test_tensor):
    test_array = test_tensor.cpu().numpy()
    true_label_mat = label_binarize(test_array, classes=[0,1,2])
    pred_prob = torch.softmax(torch.cat(test_prob),dim=1)
    _,pred = torch.max(pred_prob,1)
    pred_prob = pred_prob.cpu().detach().numpy()
    new_pred = pred.cpu().numpy()
    
    precision = precision_score(test_array, new_pred,average='macro')
    recall = recall_score(test_array,new_pred,average='macro')
    f1 = f1_score(test_array, new_pred, average='macro')
    accuracy = accuracy_score(test_array, new_pred)
    auc = roc_auc_score(true_label_mat,pred_prob,average='macro')
    print('Macro Result: acc:{}, precision:{}, recall:{}, f1:{}, AUC:{} '.format(accuracy,precision,recall,f1, auc))

def get_metrics(test_prob,test_tensor):
    test_array = test_tensor.cpu().numpy()
    true_label_mat = label_binarize(test_array, classes=[0,1,2])
    pred_prob = torch.softmax(torch.cat(test_prob),dim=1)
    _,pred = torch.max(pred_prob,1)
    pred_prob = pred_prob.cpu().detach().numpy()
    new_pred = pred.cpu().numpy()

    accuracy = accuracy_score(test_array, new_pred)
    micro_f1 = f1_score(test_array, new_pred, average='weighted')
    micro_auc = roc_auc_score(true_label_mat,pred_prob,average='micro')
    macro_f1 = f1_score(test_array, new_pred, average='macro')
    macro_auc = roc_auc_score(true_label_mat,pred_prob,average='macro')
    return accuracy, micro_f1, micro_auc, macro_f1, macro_auc

def cal_performence(pred, gold, class_weight=None, smoothing=False):
    '''apply label smooth if needed'''
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, gold)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.eq(gold).sum().item()
    return loss, n_correct

def train_epoch(model, train_data, optimizer, device,scheduler=None):
    '''Epoch operation in training phase'''
    model.train()
    total_loss = 0
    n_total = 0
    total_correct_label=0
    
    for batch in tqdm(train_data, mininterval=2, desc = ' -(Training) ', leave=False):
        left_train, right_train, train_label = map(lambda x: x.to(device), batch)
        
        optimizer.zero_grad()
        pred = model(left_train, right_train)
        
        loss, n_correct_label = cal_performence(pred, train_label)
        
        loss.backward()
        
        # optimizer.step_and_update_lr()
        optimizer.step()
        # scheduler.step()
        
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

            pred = model(left_dev, right_dev)

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
            input_left, input_right, test_label = map(lambda x: x.to(device), batch)

            pred = model(input_left, input_right)

            loss, n_correct_label = cal_performence(pred, test_label)

            # note keeping
            total_loss += loss.item()
            total_correct_label += n_correct_label
            n_total += len(test_label)
            test_predict_prob.append(pred.cpu())
            test_pred_label.append(test_label.cpu())
    test_pred_label = torch.cat(test_pred_label)

    accuracy, micro_f1, micro_auc, macro_f1, macro_auc = get_metrics(test_predict_prob,test_pred_label)
    accuracy_label = total_correct_label/n_total
    loss_per_sample = total_loss / n_total
    return accuracy_label, loss_per_sample,accuracy, micro_f1, micro_auc, macro_f1, macro_auc


def train(model, training_data, validation_data, test_data_ans, test_data_que, test_data_dom, optimizer, device, opt, scheduler):
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
            model, training_data, optimizer, device, scheduler
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
            print('[Info]: For aug question task:')
            test_accu_que, test_loss_que, accuracy_que, micro_f1_que, micro_auc_que, macro_f1_que, macro_auc_que = test_epoch(model,test_data_que,device)
            print('test acc:{}, micro f1:{}, micro auc:{}, macro f1:{}, macro auc:{}'.format(accuracy_que, micro_f1_que, micro_auc_que, macro_f1_que, macro_auc_que))
            # print('[Info]: For question task New Test accuracy {}'.format(test_accu_que))
            print('[Info]: For aug domain task:')
            test_accu_dom, test_loss_dom, accuracy_dom, micro_f1_dom, micro_auc_dom, macro_f1_dom, macro_auc_dom = test_epoch(model,test_data_dom,device)
            print('test acc:{}, micro f1:{}, micro auc:{}, macro f1:{}, macro auc:{}'.format(accuracy_dom, micro_f1_dom, micro_auc_dom, macro_f1_dom, macro_auc_dom))

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

        #             # test_accu, test_loss = eval_epoch(model,test_data_que, device)
        #             # print('[Info]: For question task New Test accuracy {}'.format(test_accu))
        #             # test_accu, test_loss = eval_epoch(model,test_data_ans, device)
        #             # print('[Info]: For answer task New Test accuracy {}'.format(test_accu))
        #             # test_accu, test_loss = eval_epoch(model,test_data_dom, device)
        #             # print('[Info]: For domain task New Test accuracy {}'.format(test_accu))
        
        # if log_train_file and log_valid_file:
        #     with open(log_train_file,'a') as log_tf, open(log_valid_file,'a') as log_vf:
        #         log_tf.write('{epoch}, {loss: 8.5f},{accu_label: 3.3f}\n'.format(
        #             epoch = epoch_i, loss=train_loss, accu_label=100*train_label_accu
        #         ))
        #         log_vf.write('{epoch}, {loss: 8.5f},{accu_label: 3.3f}\n'.format(
        #             epoch = epoch_i, loss=valid_loss, accu_label=100*valid_accu
        #         ))
    return accuracy_que, micro_f1_que, micro_auc_que, macro_f1_que, macro_auc_que, accuracy_dom, micro_f1_dom, micro_auc_dom, macro_f1_dom, macro_auc_dom

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size',type=int, default=64)
    
    parser.add_argument('-d_model',type=int, default=128)
    parser.add_argument('-number_block',type=int, default=4)
    parser.add_argument('-head_number',type=int, default=8)
    parser.add_argument('-d_ff',type=int, default=1024)
    parser.add_argument('-class_num',type=int, default=3)
    parser.add_argument('-seq_len',type=int, default=30)

    parser.add_argument('-n_warmup_steps',type=int, default=2000)
    
    parser.add_argument('-dropout',type=float,default=0.2)
    parser.add_argument('-emb_dropout',type=float,default=0.1)
    
    parser.add_argument('-log',type=str, default='./logs/new_model_3way_que')
    parser.add_argument('-save_model',type=str, default='./saved_model/new_model_3way_que')
    parser.add_argument('-save_mode', type=str, choices=['all','best'], default='best')

    opt = parser.parse_args()
    print(opt)

    accuracys_que, f1s_que_micro, aucs_que_mirco, f1s_que_macro, aucs_que_macro = [],[],[],[],[]
    accuracys_dom, f1s_dom_micro, aucs_dom_mirco, f1s_dom_macro, aucs_dom_macro = [],[],[],[],[]
    # for i in range(2, 3):
    seed_num = 1000
    torch.manual_seed(seed_num)
    batch_size=opt.batch_size

    # load preprocessed data
    test_left_question = torch.LongTensor(torch.load("./semeval/processed_data/3way/unseen_questions/left_test_question"))
    test_right_question = torch.LongTensor(torch.load("./semeval/processed_data/3way/unseen_questions/right_test_question"))
    # answer for test data
    test_left_answer = torch.LongTensor(torch.load("./semeval/processed_data/3way/unseen_answers/left_test_answer"))
    test_right_answer = torch.LongTensor(torch.load("./semeval/processed_data/3way/unseen_answers/right_test_answer"))
    # domain for test data
    test_left_dom = torch.LongTensor(torch.load('./semeval/processed_data/3way/unseen_domains/left_test_domain'))
    test_right_dom = torch.LongTensor(torch.load('./semeval/processed_data/3way/unseen_domains/right_test_domain'))
    #yaoyongde train data
    train_data_left = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/3way/left_aug_train'))
    train_data_right = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/3way/right_aug_train'))
    #yaoyongde dev data
    dev_data_left = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/3way/left_aug_dev'))
    dev_data_right = torch.LongTensor(torch.load('./semeval/sem_new_train_dev/3way/right_aug_dev'))

    #yaoyongde label
    train_label = torch.LongTensor([i-1 for i in torch.load('./semeval/sem_new_train_dev/3way/train_aug_label')])
    dev_label = torch.LongTensor([i-1 for i in torch.load('./semeval/sem_new_train_dev/3way/dev_aug_label')])
    test_label_question = torch.LongTensor([i-1 for i in torch.load('./semeval/processed_data/3way/unseen_questions/test_label_question')])
    test_label_answer = torch.LongTensor([i-1 for i in torch.load('./semeval/processed_data/3way/unseen_answers/test_label_answer')])
    test_label_domain = torch.LongTensor([i-1 for i in torch.load('./semeval/processed_data/3way/unseen_domains/test_label_domain')])

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_data_left, train_data_right, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(dev_data_left, dev_data_right, dev_label)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our answer test set.
    test_loader_ans = TensorDataset(test_left_answer, test_right_answer, test_label_answer)
    test_sampler_ans = SequentialSampler(test_loader_ans)
    test_dataloader_ans = DataLoader(test_loader_ans, sampler=test_sampler_ans, batch_size=batch_size)
    # Create the DataLoader for our question test set.
    test_loader_que = TensorDataset(test_left_question, test_right_question,test_label_question)
    test_sampler_que = SequentialSampler(test_loader_que)
    test_dataloader_que = DataLoader(test_loader_que, sampler=test_sampler_que, batch_size=batch_size)
    # Create the DataLoader for our domain test set.
    test_loader_dom = TensorDataset(test_left_dom, test_right_dom, test_label_domain)
    test_sampler_dom = SequentialSampler(test_loader_dom)
    test_dataloader_dom = DataLoader(test_loader_dom, sampler=test_sampler_dom, batch_size=batch_size)

    vocabulary = torch.load('./semeval/processed_data/2way/word_vocab')
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.vocab_size = len(vocabulary)

    model = CMAN_plus(
        d_model=opt.d_model, 
        number_block_1=opt.number_block, 
        head_number=opt.head_number, 
        d_ff=opt.d_ff, 
        seq_len=opt.seq_len,
        vocab_size=opt.vocab_size,
        class_num = opt.class_num,
        drop_out=opt.dropout,
        emb_dropout = opt.emb_dropout).to(opt.device)

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         filter(lambda x: x.requires_grad, model.parameters()),
    #         betas=(0.9,0.98),eps=1e-09),
    #     opt.d_model, opt.n_warmup_steps
    # )
    # total_steps = len(train_dataloader) * opt.epoch
    # optimizer = AdamW(model.parameters(),
    #           lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #           eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    #         )
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=500, 
    #     num_training_steps=total_steps
    # )
    scheduler = None
    optimizer = optim.Adadelta(model.parameters(),lr=1,rho=0.9,eps=1e-6,weight_decay=0)
    accuracy_que, micro_f1_que, micro_auc_que, macro_f1_que, macro_auc_que, accuracy_dom, micro_f1_dom, micro_auc_dom, macro_f1_dom, macro_auc_dom = train(model, train_dataloader, validation_dataloader, test_dataloader_ans, test_dataloader_que, test_dataloader_dom,optimizer, opt.device, opt,scheduler)
    
    accuracys_que.append(accuracy_que)
    f1s_que_micro.append(micro_f1_que)
    aucs_que_mirco.append(micro_auc_que)
    f1s_que_macro.append(macro_f1_que)
    aucs_que_macro.append(macro_auc_que)

    accuracys_dom.append(accuracy_dom)
    f1s_dom_micro.append(micro_f1_dom)
    aucs_dom_mirco.append(micro_auc_dom)
    f1s_dom_macro.append(macro_f1_dom)
    aucs_dom_macro.append(macro_auc_dom)
    print('For question')
    print('final average accuracy: {}'.format(np.mean(accuracys_que)))
    print('[Micro] final average f1: {}'.format(np.mean(f1s_que_micro)))
    print('[Micro] final average auc: {}'.format(np.mean(aucs_que_mirco)))
    print('[Macro] final average f1: {}'.format(np.mean(f1s_que_macro)))
    print('[Macro] final average auc: {}'.format(np.mean(aucs_que_macro)))

    print('For domain')
    print('final average accuracy: {}'.format(np.mean(accuracys_dom)))
    print('[Micro] final average f1: {}'.format(np.mean(f1s_dom_micro)))
    print('[Micro] final average auc: {}'.format(np.mean(aucs_dom_mirco)))
    print('[Macro] final average f1: {}'.format(np.mean(f1s_dom_macro)))
    print('[Macro] final average auc: {}'.format(np.mean(aucs_dom_macro)))


if __name__ == "__main__":
    main()








