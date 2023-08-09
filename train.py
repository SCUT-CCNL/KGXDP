import torch

print(torch.cuda.device_count())

import random
from metrics import evaluate_codes, evaluate_hf
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from modeling.modeling_kgxdp import *
from utils.optimization_utils import OPTIMIZER_CLASSES

from utils.parser_utils import *
import os

task_name = 'm' 
pre_train_model = 'biolink' 
task_conf = {
    'm': {
        'dropout': 0.3, 
        'output_size': 4880, 
        'evaluate_fn': evaluate_codes,
        'lr': {
            'init_lr': 0.01,
            'milestones': [20, 30],
            'dropouti': 0,
            'dropoutg': 0.3, 
            'dropoutf': 0.4, 
        },
        'dlr': 1e-3,
        'encoder_lr': 1e-3,
        'biolink':{
            'encoder': 'michiyasunaga/BioLinkBERT-base',
            'ent_emb': 'data/ddb/ent_emb.npy'
        },
        'spanbert':{
            'encoder': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
            'ent_emb': 'data/ddb/ent_emb_copy.npy'
        },
    },
    'h': {
        'dropout': 0.2, 
        'output_size': 1,
        'evaluate_fn': evaluate_hf,
        'lr': {
            'init_lr': 0.01,
            'milestones': [2, 3, 20],
            'dropouti':0.2, 
            'dropoutg': 0.2,
            'dropoutf': 0.9,
        },
        'dlr': 1e-4,
        'encoder_lr': 1e-4,
        'biolink':{
            'encoder': 'michiyasunaga/BioLinkBERT-base',
            'ent_emb': 'data/ddb/ent_emb.npy'
        },
        'spanbert':{
            'encoder': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
            'ent_emb': 'data/ddb/ent_emb_copy.npy'
        },
    },
}



def main():
    
    
    parser = get_parser() 
    args, _ = parser.parse_known_args() 
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/qagnn/', help='model output directory') 
    parser.add_argument('--save_model', default=False, dest='save_model', action='store_true') 
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--num_relation', default=34, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/mimic/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/mimic/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/mimic/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads') 
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers') 
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units') 
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers') 
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--dropouti', type=float, default=task_conf[task_name]['lr']['dropouti'], help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=task_conf[task_name]['lr']['dropoutg'], help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=task_conf[task_name]['lr']['dropoutf'], help='dropout for fully-connected layers')
    parser.add_argument('-dlr', '--decoder_lr', default=task_conf[task_name]['dlr'], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=32, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=16, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=True, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args,_ = parser.parse_known_args()
    args.batch_size = 16
    args.cuda = True 
    args.dataset = 'mimic' # mimic iv
    args.debug = False
    args.dev_statements = f'data/{args.dataset}/statement/dev.statement.jsonl'
    args.train_statements = f'data/{args.dataset}/statement/train.statement.jsonl'
    args.test_statements = f'data/{args.dataset}/statement/test.statement.jsonl'
    args.encoder = task_conf[task_name][pre_train_model]['encoder']
    args.encoder_layer = -1  
    args.encoder_lr = task_conf[task_name]['encoder_lr']  
    args.ent_emb = [task_conf[task_name][pre_train_model]['ent_emb']]
    args.inhouse = False  
    
    args.log_interval = 10 
    args.loss = 'BEC'
    args.lr_schedule = 'warmup_constant'
    args.max_epochs_before_stop = 10
    args.max_grad_norm = 1.0  
    args.max_seq_len = 20  
    args.n_epochs = 15
    args.optim = 'radam'  
    args.seed = 0
    args.warmup_steps = 200
    args.pred_dim = task_conf[task_name]['output_size']
    args.n_diagnosis_codes = 4880
    
    args.dropout_rate = task_conf[task_name]['dropout'] 
    args.hita_layers = 1

    if args.simple:
        parser.set_defaults(k=1)
    
    
    args.fp16 = False
    args.hita_config = {
    'n_diagnosis_codes': args.n_diagnosis_codes,
    'batch_size': args.batch_size,
    'num_layers': args.hita_layers,
    'dropout_rate': args.dropout_rate
    }

    if args.mode == 'train':
        train(args)

def train(args):

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')  
    export_config(args, config_path) 
    check_path(model_path) 
    
    with open(log_path, 'w') as fout:
        fout.write('step, dev_acc, test_acc\n')
    semd_emb = [np.load(path) for path in args.ent_emb]
    semd_emb = torch.tensor(np.concatenate(semd_emb, 1), dtype=torch.float)

    concept_num, concept_dim = semd_emb.size(0), semd_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    print('| dimension_concepts: {} |'.format(concept_dim))
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:0")

    print('| device0: {} |'.format(device0))
    print('| device1: {} |'.format(device1))

    
    dataset = LM_QAGNN_DataLoader(args, args.train_statements, args.train_adj,
                                        args.dev_statements, args.dev_adj,
                                        args.test_statements, args.test_adj,
                                        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                        device=(device0, device1),
                                        model_name=args.encoder, 
                                        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                        subsample=args.subsample, use_cache=args.use_cache)

    model = LM_QAGNN(args, args.pred_dim, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, n_concept=concept_num,
                            concept_dim=args.gnn_dim,
                            concept_in_dim=concept_dim,
                            n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                            p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                            pretrained_concept_emb=semd_emb, freeze_ent_emb=args.freeze_ent_emb,
                            init_range=args.init_range,
                            encoder_config={}, hita_config=args.hita_config)

    if args.load_model_path:
        model = torch.load(args.load_model_path)
        model.eval()

        evaluate_fn = task_conf[task_name]['evaluate_fn']
        if task_name == 'm':
            test_f1, test_recall = evaluate_fn(dataset.test(), model, 'test')
            best_test_recall_10 = test_recall[0]
            best_test_recall_20 = test_recall[1]
            print('test_f1_score: %.8f --- top_k_recall: %.4f, %.4f' %
                  (test_f1, best_test_recall_10, best_test_recall_20))
        elif task_name == 'h':
            test_f1_score_, test_auc = evaluate_fn(dataset.test(), model, 'test')
            print('test_f1_score: %.8f --- auc: %.4f' %
                  (test_f1_score_, test_auc))
        exit()

    model.encoder_HITA.to(device0)
    model.encoder_PreTrain.to(device0)
    model.decoder.to(device1)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder_HITA.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder_HITA.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder_PreTrain.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder_PreTrain.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    
    
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters) 

    
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    
    for name, param in model.decoder.named_parameters():
        
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    
    if args.loss == 'margin_rank': 
        
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn = torch.nn.BCELoss()

    
    def compute_loss(logits, labels):
        labels = labels.float()
        if args.loss == 'margin_rank':
            num_choice = logits.size(1)
            flat_logits = logits.view(-1)
            correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  
            correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  
            wrong_logits = flat_logits[correct_mask == 0]
            y = wrong_logits.new_ones((wrong_logits.size(0),))
            loss = loss_func(correct_logits, wrong_logits, y)  
        elif args.loss == 'cross_entropy':
            loss = loss_func(logits, labels)
        else:
            loss = loss_fn(logits, labels)
        return loss


    print()
    if args.fp16:
        print ('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    best_dev_f1, best_dev_recall, best_test_f1, best_test_recall, best_test_auc, best_dev_auc, best_epoch_id = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    start_time = time.time()
    model.train()

    for epoch_id in range(args.n_epochs): 
        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.encoder_PreTrain)
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.encoder_PreTrain)
        model.train()
        
        for qids, HF_labels, Diag_labels, \
            diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,\
            *input_data in tqdm(dataset.train(), desc='train: epoch-'+str(epoch_id)):

            optimizer.zero_grad()
            bs = HF_labels.size(0) 
            if task_name == 'm':
                labels = Diag_labels
            else:
                labels = HF_labels
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                
                if args.fp16:
                    
                    with torch.cuda.amp.autocast():
                        logits, _, (edge_idx, edge_weight), visit_att, self_att = model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,
                                          *[x[a:b] for x in input_data], layer_id=args.encoder_layer)  
                        loss = compute_loss(logits.squeeze(), labels[a:b])
                else:
                    logits, _, (edge_idx, edge_weight), visit_att, self_att = model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,
                                          *[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                    loss = compute_loss(logits.squeeze(), labels[a:b])
                
                loss = loss * (b - a) / bs
                
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
            
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            
            
            if args.max_grad_norm > 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()

            
            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                total_loss = 0
                start_time = time.time()
            global_step += 1

        
        model.eval()
        evaluate_fn = task_conf[task_name]['evaluate_fn']
        if task_name == 'm':
            
            test_f1, test_recall = evaluate_fn(dataset.test(), model, 'test')
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                
                best_epoch_id = epoch_id
            
                best_test_recall = test_recall[0]
                best_test_recall_10 = test_recall[0]
                best_test_recall_20 = test_recall[1]

            torch.save(model, './saved_models/'+str(epoch_id)+'.pt')
            print('best epoch:', str(best_epoch_id))

            print('test_f1_score: %.8f --- top_k_recall: %.4f, %.4f' %
                  (best_test_f1, best_test_recall_10, best_test_recall_20))
        elif task_name == 'h':
            
            test_f1_score_, test_auc = evaluate_fn(dataset.test(), model, 'test')
            if best_test_f1 < test_f1_score_:
                best_epoch_id = epoch_id
                best_test_f1 = test_f1_score_
                
                best_test_auc = test_auc
            print('best epoch:', str(best_epoch_id))
            print('test_f1_score: %.8f --- auc: %.4f' %
                  (best_test_f1, best_test_auc))
if __name__ == '__main__':
    main()
