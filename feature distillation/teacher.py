# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:54:49 2021

@author: Jason
"""
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score
from load_data import load_data
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from parameter import parse_args

from model import BERT_MLM

torch.cuda.empty_cache()  # 清除GPU缓存
args = parse_args()  # 加载参数


# 设置随机数种子
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)


setup_seed(args.seed)

# load BertForMasked model
model_name = '../PLMs/BERT/BertForMaskedLM/bert-base-uncased'
# template_new_tokens = ['[V1]', '[V2]', '[V3]', '[V4]', '[V5]', '[V6]', '[V7]', '[V8]', '[V9]', '[V10]']
template_new_tokens = ['[V1]', '[V2]', '[V3]', '[V4]', '[V5]', '[V6]']
# answer_new_tokens = ['[A1]', '[A2]', '[A3]', '[A4]', '[A5]', '[A6]', '[A7]', '[A8]', '[A9]', '[A10]',
#                      '[A11]', '[A12]', '[A13]', '[A14]', '[A15]', '[A16]']
# answer_new_tokens = ['[A1]', '[A2]', '[A3]', '[A4]', '[A5]', '[A6]', '[A7]', '[A8]', '[A9]', '[A10]',
#                      '[A11]', '[A12]', '[A13]', '[A14]', '[A15]', '[A16]', '[A17]', '[A18]', '[A19]']
answer_new_tokens = ['[A1]', '[A2]', '[A3]', '[A4]', '[A5]', '[A6]', '[A7]', '[A8]', '[A9]', '[A10]', '[A11]',
                     '[A12]', '[A13]', '[A14]', '[A15]', '[A16]', '[A17]', '[A18]', '[A19]', '[A20]', '[A21]']

connective_new_tokens = []
for i in range(args.conn_num):
    connective_new_tokens.append('[C%d]'%(i+1))

tokenizer = BertTokenizer.from_pretrained(model_name)
original_vocab_size = len(list(tokenizer.get_vocab()))
tokenizer.add_tokens(template_new_tokens)
vocab_size_after_template = len(tokenizer)
print('Vocabulary size of tokenizer after adding template new tokens : %d'%vocab_size_after_template)
tokenizer.add_tokens(answer_new_tokens)
vocab_size_after_answer = len(tokenizer)
print('Vocabulary size of tokenizer after adding answer new tokens : %d'%vocab_size_after_answer)
tokenizer.add_tokens(connective_new_tokens)
vocab_size_after_connective = len(tokenizer)
print('Vocabulary size of tokenizer after adding connective new tokens : %d'%vocab_size_after_connective)

# load data tsv file
train_data, dev_data, test_data = load_data()

# get arg_1 arg_2 label from data
train_arg_1, train_arg_2, train_label, train_conn_label, train_conn, train_subtype_list = prepro_data_train(train_data)
dev_arg_1, dev_arg_2, dev_label, dev_conn_label, dev_conn, dev_subtype_list = prepro_data_dev(dev_data)
test_arg_1, test_arg_2, test_label, test_conn_label, test_conn, test_subtype_list = prepro_data_test(test_data)
# print(len(train_arg_1))

# train_sample_set = []
# for i in range(len(answer_new_tokens)):
#     train_sample_set.append([])
# for i in range(len(train_subtype_list)):
#     train_sample_set[train_subtype_list[i]].append(i)

train_subt_list = torch.LongTensor(train_subtype_list)
label_tr = torch.LongTensor(train_label)
label_de = torch.LongTensor(dev_label)
label_te = torch.LongTensor(test_label)
print('Data loaded')


def arg_1_prepro(arg_1):
    arg_1_new = []
    for each_string in arg_1:
        encode_dict = tokenizer.encode_plus(
            each_string,
            add_special_tokens=False,
            padding='max_length',
            max_length=args.arg1_len,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt')
        decode_input = tokenizer.decode(encode_dict['input_ids'][0]).replace(' [PAD]', '')
        arg_1_new.append(decode_input)
    return arg_1_new


train_arg_1 = arg_1_prepro(train_arg_1)
dev_arg_1 = arg_1_prepro(dev_arg_1)
test_arg_1 = arg_1_prepro(test_arg_1)


def get_batch(text_data1, text_data2, conn_label, indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # 一个batch内，每个input_ids中[MASK]的位置
    if len(template_new_tokens) == 0:
        print('new token error')
    after_sub_conn_id = original_vocab_size + len(template_new_tokens) / 2

    for idx in indices:
        encode_dict = tokenizer.encode_plus(
            # text_data1[idx] + ' [V1]' + ' [MASK] ' + '[V2] ' + text_data2[idx],
            text_data1[idx] + ' [V1] [V2] [V3] ' + connective_new_tokens[conn_label[idx]] + ' [V4] [V5] [V6] ' + text_data2[idx],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        try: mask_indices.append(np.argwhere(np.array(encode_dict['input_ids']) == after_sub_conn_id)[0][1] - 1)
        except IndexError:
            print(encode_dict['input_ids'])
            # print(np.argwhere(np.array(encode_dict['input_ids']) == 103))

    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)

    return batch_ids, batch_mask, mask_indices

# centroid = torch.load("./train_centroid_BERT.pth")

# map_16_to_4 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]

# map_19_to_4 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]


map_21_to_4 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]

# ---------- network ----------
net = BERT_MLM(args).cuda()
ebd = net.BERT_MLM.resize_token_embeddings(len(tokenizer))

# with open('answer_vecs_16.npy', 'rb') as f:
#     vs = np.load(f)

with open('answer_vecs_21.npy', 'rb') as f:
    vs = np.load(f)

with torch.no_grad():
    net.base_model.embeddings.word_embeddings.weight[vocab_size_after_template:vocab_size_after_answer] = torch.Tensor(vs)

with open('connective_vecs_181.npy', 'rb') as f:
    vs = np.load(f)

with torch.no_grad():
    net.base_model.embeddings.word_embeddings.weight[vocab_size_after_answer:] = torch.Tensor(vs)




# AdamW
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)

# criterion_1 = nn.MSELoss(reduction='none').cuda()
criterion = nn.CrossEntropyLoss().cuda()

# creat file to save model and result
# file_out = open("./out.txt", "w")
file_out = open('./' + args.file_out + '.txt', "w")
# if not os.path.exists('net_model'):
# os.mkdir('net_model')

print('epoch_num:', args.num_epoch)
print('epoch_num:', args.num_epoch, file=file_out)
print('wd:', args.wd)
print('wd:', args.wd, file=file_out)
print('initial_lr:', args.lr)
print('initial_lr:', args.lr, file=file_out)

# total_steps = (args.train_size // args.batch_size + 1) * args.num_epoch
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warm_ratio * total_steps, num_training_steps = total_steps)

##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch + 1)
    print('Epoch: ', epoch + 1, file=file_out)
    all_indices = torch.randperm(args.train_size).split(args.batch_size)
    #print(all_indices)
    loss_epoch = 0.0
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    start = time.time()

    # print('lr:', scheduler.get_last_lr())
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], file=file_out)


    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    for i, batch_indices in enumerate(all_indices, 1):
        # torch.autograd.set_detect_anomaly(True)
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices = get_batch(train_arg_1, train_arg_2, train_conn_label, batch_indices)

        # batch_arg2, mask_arg2 = get_batch(train_arg_2, batch_indices)
        batch_arg = batch_arg.cuda()
        # batch_arg2 = batch_arg2.cuda()
        mask_arg = mask_arg.cuda()
        # mask_arg2 = mask_arg2.cuda()

        y = Variable(label_tr[batch_indices]).cuda()
        y_subtype = train_subt_list[batch_indices].cuda()

        # fed data into network
        # print(batch_arg1, mask_arg1, batch_arg2, mask_arg2)
        out_sense, out_ans, mask_ebd = net(batch_arg, mask_arg, token_mask_indices,
                                           vocab_size_after_template, vocab_size_after_answer, map_21_to_4)

        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

        # loss
        loss = criterion(out_ans, y_subtype)
        # if args.if_con in ['POS', 'NEG', 'ALL']:
        #     loss = criterion(out_ans, y_subtype) - args.aux_factor * aux_loss
        # else:
        #     loss = criterion(out_ans, y_subtype)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # report
        loss_epoch += loss.item()
        if i % (3000 // args.batch_size) == 0:
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                             average='macro')), file=file_out)
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                             average='macro')))
            loss_epoch = 0.0
            acc = 0.0
            f1_pred = torch.IntTensor([]).cuda()
            f1_truth = torch.IntTensor([]).cuda()
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.randperm(args.dev_size).split(args.batch_size)
    loss_epoch = []
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    # dev_ap = []
    net.eval()
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices = get_batch(dev_arg_1, dev_arg_2, dev_conn_label, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_de[batch_indices]).cuda()

        # fed data into network
        with torch.no_grad():
            out_sense, out_ans, _ = net(batch_arg, mask_arg, token_mask_indices,
                                        vocab_size_after_template, vocab_size_after_answer, map_21_to_4)


        # pred = my_verbalizer(out).cuda()

        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

        # loss
        loss_epoch.extend([loss.item()])

    # report
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    print('Dev Loss={:.4f}, Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(loss_epoch, acc / args.dev_size,
                                                                        f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                 average='macro')), file=file_out)
    print('Dev Loss={:.4f}, Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(loss_epoch, acc / args.dev_size,
                                                                        f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                 average='macro')))

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.randperm(args.test_size).split(args.batch_size)
    loss_epoch = []
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    net.eval()
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices = get_batch(test_arg_1, test_arg_2, test_conn_label, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_te[batch_indices]).cuda()

        # fed data into network
        with torch.no_grad():
            out_sense, out_ans, _ = net(batch_arg, mask_arg, token_mask_indices,
                                        vocab_size_after_template, vocab_size_after_answer, map_21_to_4)

        # out_sense, out_ans, _ = net(batch_arg, mask_arg, token_mask_indices, vocab_size_after_template, map_16_to_4)

        # pred = my_verbalizer(out).cuda()

        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

        # loss
        loss_epoch.extend([loss.item()])

    # report
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    print('Test Loss={:.4f}, Test Acc={:.4f}, Test F1_score={:.4f}'.format(loss_epoch, acc / args.test_size,
                                                                           f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')), file=file_out)
    print('Test Loss={:.4f}, Test Acc={:.4f}, Test F1_score={:.4f}'.format(loss_epoch, acc / args.test_size,
                                                                           f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')))

    # torch.save(net.state_dict(), './net_model/epoch_'+str(epoch+1)+'.pkl')

    ############################################################################
    ##################################  save  ##################################
    ############################################################################

    # 保存最佳教师模型的结果

    if epoch == 2:
        out_tea_Response = torch.zeros(args.train_size, len(map_21_to_4)).cuda()
        out_tea_Feature = torch.zeros(args.train_size, args.h_dim).cuda()
        all_indices = torch.randperm(args.train_size).split(args.batch_size)
        net.eval()
        for batch_indices in all_indices:
            batch_arg, mask_arg, token_mask_indices = get_batch(train_arg_1, train_arg_2, train_conn_label, batch_indices)
            batch_arg = batch_arg.cuda()
            mask_arg = mask_arg.cuda()
            with torch.no_grad():
                _, out_ans, last_hidden_states = net(batch_arg, mask_arg, token_mask_indices,
                                                     vocab_size_after_template, vocab_size_after_answer, map_21_to_4)

            out_tea_Response[batch_indices] = out_ans
            out_tea_Feature[batch_indices] = last_hidden_states
        Teacher_Response = out_tea_Response.detach().cpu().numpy()
        Teacher_Feature = out_tea_Feature.detach().cpu().numpy()
        with open('./out_teacher_BERT_Response.npy', 'wb') as f_1:
            np.save(f_1, Teacher_Response)

        with open('./out_teacher_BERT_Feature.npy', 'wb') as f_2:
            np.save(f_2, Teacher_Feature)
        print('teacher results saved !')

        # with open('./out_teacher_BERT_Response.npy', 'rb') as f:
        #     otea1 = np.load(f)
        #
        # with open('./out_teacher_BERT_Feature.npy', 'rb') as f:
        #     otea2 = np.load(f)
        #
        # OUT_Tea = torch.Tensor(otea1)


file_out.close()
