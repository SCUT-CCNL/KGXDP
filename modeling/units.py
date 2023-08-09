import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy

def load_data(training_file, validation_file, testing_file):
    train = np.array(pickle.load(open(training_file, 'rb')))
    validate = np.array(pickle.load(open(validation_file, 'rb')))
    test = np.array(pickle.load(open(testing_file, 'rb')))
    return train, validate, test

def cut_data(training_file, validation_file, testing_file):
    train = list(pickle.load(open(training_file, 'rb')))
    validate = list(pickle.load(open(validation_file, 'rb')))
    test = list(pickle.load(open(testing_file, 'rb')))
    for dataset in [train, validate, test]:
        dataset[0] = dataset[0][0: len(dataset[0]) // 18]
        dataset[1] = dataset[1][0: len(dataset[1]) // 18]
        dataset[2] = dataset[2][0: len(dataset[2]) // 18]
    return train, validate, test


def pad_time_hita(seq_time_step, options):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths) 
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)

    return seq_time_step

def pad_time(seq_time_step, max_len):
    lengths = np.array([len(seq) for seq in seq_time_step])
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < max_len:
            seq_time_step[k].append(100000)

    return seq_time_step

def pad_matrix_new(seq_diagnosis_codes, n_diagnosis, maxlen):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = n_diagnosis
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1


    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]-1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1


    return batch_diagnosis_codes, batch_mask, batch_mask_final, batch_mask_code


def calculate_cost_tran(model, data, options, max_len, loss_function=F.cross_entropy):
    model.eval()
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0

    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, options['n_diagnosis_codes'])
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        logit, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)
        loss = loss_function(logit, labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes):

    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes])
    return batch_diagnosis_codes, batch_time_step


def adjust_input_hita(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes, batch_time_step2):
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
            batch_time_step2[ind] = batch_time_step2[ind][-(max_len):]
        batch_time_step[ind].append(0)
        batch_time_step2[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes])
    return batch_diagnosis_codes, batch_time_step, batch_time_step2

def adjust_input_pretrain(batch_diagnosis_codes, max_len, n_diagnosis_codes):
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
        batch_diagnosis_codes[ind].append([n_diagnosis_codes])
    return batch_diagnosis_codes

def pad_path(paths, max_num_path):
    this_paths_len = len(paths)
    path_mask = np.zeros(max_num_path)
    pad_txt = 'Urine retention may cause Renal failure'
    if len(paths) < max_num_path:
        for i in range(max_num_path - this_paths_len):
            paths.append(pad_txt)
            path_mask[this_paths_len+i] = 1
    else:
        paths = paths[:max_num_path]
    return paths, path_mask


def pad_matrix_new(seq_diagnosis_codes, n_diagnosis, maxlen):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = n_diagnosis
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)  
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)  
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)  

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1

    for i in range(n_samples):
        batch_mask[i, 0:lengths[i] - 1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    return batch_diagnosis_codes, batch_mask, batch_mask_final, batch_mask_code

def pad_matrix_pretrain(seq_diagnosis_codes, n_diagnosis, maxlen, mask_prob=0.15):
    mask_value = 1000000
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = n_diagnosis
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32) 
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32) 
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32) 

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1


    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]-1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_diagnosis_codes_copy = batch_diagnosis_codes.copy()
    masked_batch, mask, _ = \
        mask_diagnosis_codes(batch_diagnosis_codes_copy, mask_prob, n_diagnosis_codes, mask_value)
    return batch_diagnosis_codes, batch_mask, batch_mask_final, batch_mask_code, \
        masked_batch, mask

def mask_diagnosis_codes(batch_diagnosis_codes, mask_prob, pad_value, mask_value):
    batch_diagnosis_codes = torch.tensor(batch_diagnosis_codes)
    
    num_samples, max_visit, max_codes = batch_diagnosis_codes.size()

    
    mask = torch.full(batch_diagnosis_codes.size(), False, dtype=torch.bool)

    
    for i in range(num_samples):
        for j in range(max_visit):
            for k in range(max_codes):
                if batch_diagnosis_codes[i, j, k] != pad_value and torch.rand(1) < mask_prob:
                    mask[i, j, k] = True

    
    masked_batch = batch_diagnosis_codes.masked_fill(mask, mask_value)

    return masked_batch, mask, batch_diagnosis_codes


class FocalLoss(nn.Module):
    r

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()


        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss