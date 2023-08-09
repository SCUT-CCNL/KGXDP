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
# 参数获取
from utils.parser_utils import *
import os

task_name = 'm' # choice from ['h', 'm']
pre_train_model = 'biolink' # choice from ['spanbert', 'biolink']
task_conf = {
    'm': {
        'dropout': 0.3, # 0.3
        'output_size': 4880, # 5985
        'evaluate_fn': evaluate_codes,
        'lr': {
            'init_lr': 0.01,
            'milestones': [20, 30],
            'dropouti': 0,
            'dropoutg': 0.3, # 0.3
            'dropoutf': 0.4, # 0.4
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
        'dropout': 0.2, # 0.2
        'output_size': 1,
        'evaluate_fn': evaluate_hf,
        'lr': {
            'init_lr': 0.01,
            'milestones': [2, 3, 20],
            'dropouti':0.2, # 0.2
            'dropoutg': 0.2,# 0.2
            'dropoutf': 0.9,# 0.5
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
# 评估正确率

# 主函数
def main():
    # 感觉就是参数问题，干脆直接定义参数，然后在这里跑
    # 至少还能debug，费点时间
    parser = get_parser() # 罪魁祸首，获取自定义默认参数
    args, _ = parser.parse_known_args() # 与parse_args区别，与add_argument关系

    # 额外配置参数
    # 数据处理、模型存储位置
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation') # 训练模式或评价模型
    parser.add_argument('--save_dir', default=f'./saved_models/qagnn/', help='model output directory') # 模型、log等存储路径
    parser.add_argument('--save_model', default=False, dest='save_model', action='store_true') # 是否保存模型
    # parser.add_argument('--load_model_path', default='./saved_models/3.pt') # 是否加载模型
    parser.add_argument('--load_model_path', default=None) # 是否加载模型

    # data（数据，引入个人患者子图）
    parser.add_argument('--num_relation', default=34, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/mimic/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/mimic/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/mimic/graph/test.graph.adj.pk')
    # parser.add_argument('--train_adj', default=f'data/MIMIC-IV/graph/train.graph.adj.pk')
    # parser.add_argument('--dev_adj', default=f'data/MIMIC-IV/graph/dev.graph.adj.pk')
    # parser.add_argument('--test_adj', default=f'data/MIMIC-IV/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

    # model architecture（模型结构）
    # 应该是指GNN层数，即Wk与aijk中的k数目。多层再aggregate为一层
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing') 
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads') # 注意力头，即对hj而言，aij的i数目
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers') # 图神经网络dimension
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units') # 全连接层dimension
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers') # default为0，即不设全连接层？
    # 冻结entity embedding层，指不训练
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')

    # 指读入的患者个人子图，最大节点数？
    parser.add_argument('--max_node_num', default=200, type=int)
    # 为True时，GNN只有一层，即不是Multi attention head
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    # subsample小于1.0时，会依据inhouse（内部）是否为1，决定是否取子样本训练
    parser.add_argument('--subsample', default=1.0, type=float)
    # 初始化权重，所用正态分布的标准差
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

    # regularization（m）
    parser.add_argument('--dropouti', type=float, default=task_conf[task_name]['lr']['dropouti'], help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=task_conf[task_name]['lr']['dropoutg'], help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=task_conf[task_name]['lr']['dropoutf'], help='dropout for fully-connected layers')

    # optimization（优化器）
    parser.add_argument('-dlr', '--decoder_lr', default=task_conf[task_name]['dlr'], type=float, help='learning rate') # h
    parser.add_argument('-mbs', '--mini_batch_size', default=32, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=16, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=True, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args,_ = parser.parse_known_args()
    
    # 指定参数（方便调试
    args.batch_size = 16
    args.cuda = True # 是否用cuda？在服务器上为True
    args.dataset = 'mimic' # mimic iii
    # args.dataset = 'MIMIC-IV' # mimic iv
    args.debug = False
    args.dev_statements = f'data/{args.dataset}/statement/dev.statement.jsonl'
    args.train_statements = f'data/{args.dataset}/statement/train.statement.jsonl'
    args.test_statements = f'data/{args.dataset}/statement/test.statement.jsonl'
    args.encoder = task_conf[task_name][pre_train_model]['encoder']
    args.encoder_layer = -1  # 选最后一层layer
    args.encoder_lr = task_conf[task_name]['encoder_lr']  # m
    args.ent_emb = [task_conf[task_name][pre_train_model]['ent_emb']]
    args.inhouse = False  # 直接赋了False值
    # 搞不清是啥，猜是：内部指定训练的qids。即有一个范围，排除此外的qid
    args.log_interval = 10 # log记录的间隔，10个step
    args.loss = 'BEC'
    args.lr_schedule = 'warmup_constant'
    args.max_epochs_before_stop = 10
    args.max_grad_norm = 1.0  # 最大梯度值？用于解决梯度爆炸问题
    args.max_seq_len = 20  # 最多的就诊次数
    args.n_epochs = 15
    args.optim = 'radam'  # 默认
    args.seed = 0
    args.warmup_steps = 200
    args.pred_dim = task_conf[task_name]['output_size']
    args.n_diagnosis_codes = 4880
    # args.n_diagnosis_codes = 5985
    args.dropout_rate = task_conf[task_name]['dropout'] # m # m
    args.hita_layers = 1

    if args.simple:
        parser.set_defaults(k=1)
    # 设置在某条件下，采用fp16训练
    # args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')
    args.fp16 = False
    args.hita_config = {
    'n_diagnosis_codes': args.n_diagnosis_codes,
    'batch_size': args.batch_size,
    'num_layers': args.hita_layers,
    'dropout_rate': args.dropout_rate
    }
    # 输出参数（判断是否有问题
    # print(args)

    # 训练或衡量模型
    if args.mode == 'train':
        train(args)
    # elif args.mode == 'eval_detail':
    #     # raise NotImplementedError
    #     eval_detail(args)
    # else:
    #     # 库中已定义错误
    #     raise ValueError('Invalid mode')


def train(args):

    # 随机种子的选择能够减少一定程度上算法结果的随机性，也就是更接近于原始作者的结果
    # 随机种子的确定，意味着随机数的确定（因为软件中不存在真正的随机数，都是伪随机数）
    # 主要是对后续生成随机数时，作一个固定；避免因随机数不同，导致结果变动
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed) # manual_seed，手动设置种子？
    # not available，不可用
    # if torch.cuda.is_available() and args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    # 各种配置、日志的存储路径
    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')  # 以表格形式存储log
    export_config(args, config_path) # 存储参数设置
    check_path(model_path) # 检查路径是否存在？不存在就新建
    # 写入step、dev_acc、test_acc列
    with open(log_path, 'w') as fout:
        fout.write('step, dev_acc, test_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    # 加载数据

    semd_emb = [np.load(path) for path in args.ent_emb] # 加载entity embedding
    # 化为tensor张量
    semd_emb = torch.tensor(np.concatenate(semd_emb, 1), dtype=torch.float)

    # 实体数、及每个实体的维度大小
    concept_num, concept_dim = semd_emb.size(0), semd_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    print('| dimension_concepts: {} |'.format(concept_dim))
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:0")
    # device0 = torch.device("cpu")
    # device1 = torch.device("cpu")
    # 输出设备
    print('| device0: {} |'.format(device0))
    print('| device1: {} |'.format(device1))

    # QA-GNN数据加载（训练集、验证集、测试集）
    dataset = LM_QAGNN_DataLoader(args, args.train_statements, args.train_adj,
                                        args.dev_statements, args.dev_adj,
                                        args.test_statements, args.test_adj,
                                        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                        device=(device0, device1),
                                        model_name=args.encoder, # 编码器选择
                                        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                        subsample=args.subsample, use_cache=args.use_cache)

    ###################################################################################################
    #   Build model                                                                                   #
    model = LM_QAGNN(args, args.pred_dim, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, n_concept=concept_num,
                            concept_dim=args.gnn_dim,
                            concept_in_dim=concept_dim,
                            n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                            p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                            pretrained_concept_emb=semd_emb, freeze_ent_emb=args.freeze_ent_emb,
                            init_range=args.init_range,
                            encoder_config={}, hita_config=args.hita_config)

    # default为None，是否加载已有模型
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


    # cuda
    model.encoder_HITA.to(device0)
    model.encoder_PreTrain.to(device0)
    model.decoder.to(device1)

    # no_decay中出现的参数不优化？（实际中，似乎没什么影响）
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 无weigth_decay指定，默认为0。即不设置权重衰减
    # encoder、decoder参数设置（构造了4组参数？）
    grouped_parameters = [
        {'params': [p for n, p in model.encoder_HITA.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder_HITA.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder_PreTrain.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder_PreTrain.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    # 无weigth_decay指定，默认为0。即不设置权重衰减
    # encoder、decoder参数设置（构造了4组参数？）
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters) # 优化器

    # 学习率调整方式：固定不变(fixed)，
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    # 改变学习率，先使用小学习率，然后突变为一个较大的学习率
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    # 线性增大学习率
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    # print('parameters:')
    for name, param in model.decoder.named_parameters():
        # 是否需要梯度？全连接层的神经网络默认是要的
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    # 损失计算函数设置
    if args.loss == 'margin_rank': # 间隔排序损失函数（可调偏移值）
        # 可用于：GAN、排名任务、开源实现和实例非常少
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    # 交叉熵
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn = torch.nn.BCELoss()

    # 计算损失函数
    def compute_loss(logits, labels):
        labels = labels.float()
        if args.loss == 'margin_rank':
            num_choice = logits.size(1)
            flat_logits = logits.view(-1)
            correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
            correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
            wrong_logits = flat_logits[correct_mask == 0]
            y = wrong_logits.new_ones((wrong_logits.size(0),))
            loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
        elif args.loss == 'cross_entropy':
            loss = loss_func(logits, labels)
        else:
            loss = loss_fn(logits, labels)
        return loss

    ##########################################################################
    #   Training                                                             #
    ##########################################################################

    # 训练函数
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
    # freeze是指冻结训练，一般针对预训练模型，可冻结已训练好的部分，转而训练其它部分
    # 可以提高训练效率
    # freeze_net(model.encoder_PreTrain) # 改
    for epoch_id in range(args.n_epochs): # 改
        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.encoder_PreTrain)
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.encoder_PreTrain)
        model.train()
        # 喂数据
        for qids, HF_labels, Diag_labels, \
            diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,\
            *input_data in tqdm(dataset.train(), desc='train: epoch-'+str(epoch_id)):

            optimizer.zero_grad()
            bs = HF_labels.size(0) # batch size
            if task_name == 'm':
                labels = Diag_labels
            else:
                labels = HF_labels
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                # fp16半精度，一定要上cuda吗？
                if args.fp16:
                    # 可能用不到cuda？
                    with torch.cuda.amp.autocast():
                        logits, _, (edge_idx, edge_weight), visit_att, self_att = model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,
                                          *[x[a:b] for x in input_data], layer_id=args.encoder_layer)  # 标记，出问题之处
                        loss = compute_loss(logits.squeeze(), labels[a:b])
                else:
                    logits, _, (edge_idx, edge_weight), visit_att, self_att = model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,
                                          *[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                    loss = compute_loss(logits.squeeze(), labels[a:b])
                # print(loss,'*'*30)
                loss = loss * (b - a) / bs
                # 梯度回传？
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # 总损失计算
                total_loss += loss.item()
            # 优化器，步骤
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # clip_grad_norm（梯度裁剪，即减小剃度）。
            # 只解决梯度爆炸问题，max_grad_norm越大，解决越柔和；反之，越剧烈。
            if args.max_grad_norm > 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()

            # 输出训练情况
            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                total_loss = 0
                start_time = time.time()
            global_step += 1

        # 验证模型
        model.eval()
        evaluate_fn = task_conf[task_name]['evaluate_fn']
        if task_name == 'm':
            # dev_f1, dev_recall = evaluate_fn(dataset.dev(), model, 'dev')  # 采用验证集数据
            test_f1, test_recall = evaluate_fn(dataset.test(), model, 'test')
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                # best_dev_f1 = dev_f1
                best_epoch_id = epoch_id
            # if test_recall[0] > best_test_recall:
                best_test_recall = test_recall[0]
                best_test_recall_10 = test_recall[0]
                best_test_recall_20 = test_recall[1]
            # if dev_recall[0] > best_dev_recall:
            #     best_dev_recall = dev_recall[0]
            #     best_dev_recall_10 = dev_recall[0]
            #     best_dev_recall_20 = dev_recall[1]
            torch.save(model, './saved_models/'+str(epoch_id)+'.pt')
            print('best epoch:', str(best_epoch_id))
            # print('dev_f1_score: %.8f --- top_k_recall: %.4f, %.4f' %
            #       (best_dev_f1, best_dev_recall_10, best_dev_recall_20))
            print('test_f1_score: %.8f --- top_k_recall: %.4f, %.4f' %
                  (best_test_f1, best_test_recall_10, best_test_recall_20))
        elif task_name == 'h':
            # dev_f1_score_, dev_auc = evaluate_fn(dataset.dev(), model, 'dev')
            test_f1_score_, test_auc = evaluate_fn(dataset.test(), model, 'test')
            if best_test_f1 < test_f1_score_:
                best_epoch_id = epoch_id
                best_test_f1 = test_f1_score_
                # best_dev_f1 = dev_f1_score_
            # if best_test_auc < test_auc:
                best_test_auc = test_auc
                # best_dev_auc = dev_auc
            print('best epoch:', str(best_epoch_id))
            # print('dev_f1_score: %.8f --- auc: %.4f' %
            #       (best_dev_f1, best_dev_auc))
            print('test_f1_score: %.8f --- auc: %.4f' %
                  (best_test_f1, best_test_auc))

if __name__ == '__main__':
    # 主函数
    main()