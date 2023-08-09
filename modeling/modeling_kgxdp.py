from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_hita import *
from utils.layers import *
from modeling.transformer_kgxdp import TransformerTime

class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                    dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin' 
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_etype +1 + n_ntype *2, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))


        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout


    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, return_attention_weights=True):
        all_gnn_attn = []
        all_edge_map = []
        for _ in range(self.k):
            if return_attention_weights:
                _X, (edge_idx, edge_weight) = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                
                gnn_attn = edge_weight[:, - 1]
                edge_map = edge_idx

                gnn_attn = gnn_attn[0:500]
                edge_map = edge_map[:, 0:500]
                
                

                all_gnn_attn.append(gnn_attn)
                all_edge_map.append(edge_map)
            else:
                _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)
        if return_attention_weights:
            return _X, (all_edge_map, all_gnn_attn)
        else:
            return _X


    def forward(self, H, A, node_type, node_score, cache_output=False, return_attention_weights=True):
        _batch_size, _n_nodes = node_type.size()

        
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) 

        
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) 
            js = torch.pow(1.1, js) 
            B = torch.sin(js * node_score) 
            node_score_emb = self.activation(self.emb_score(B)) 
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) 
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) 
            node_score_emb = self.activation(self.emb_score(B)) 


        X = H
        edge_index, edge_type = A 
        _X = X.view(-1, X.size(2)).contiguous() 
        _node_type = node_type.view(-1).contiguous() 
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() 

        if return_attention_weights:
            _X, (all_gnn_atten, all_edge_map) = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)
        else:
            _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1) 

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        if return_attention_weights:
            return output, (all_gnn_atten, all_edge_map)
        else:
            return output



class QAGNN(nn.Module):
    def __init__(self, args,pre_dim, k, n_ntype, n_etype, sent_dim,
                n_concept, concept_dim, concept_in_dim, n_attention_head,
                fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                pretrained_concept_emb=None, freeze_ent_emb=True,
                init_range=0, gram_dim=768):
        super().__init__()
        self.pre_dim = pre_dim
        self.init_range = init_range
        
        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                                use_contextualized=False, concept_in_dim=concept_in_dim,
                                                pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        
        self.svec2nvec = nn.Linear(sent_dim, concept_dim) 

        self.concept_dim = concept_dim

        self.activation = GELU() 
        
        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                        input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim) 

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)
        self.dropout_g = nn.Dropout(0.9) 
        self.dropout_z = nn.Dropout(0.9)

        self.activateOut = torch.nn.Sigmoid()

        if init_range > 0:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, sent_vecs, dl_vec, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None, cache_output=False, return_attention_weights=True):
        
        
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) 
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) 
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) 


        
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() 
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] 
        node_scores = node_scores.squeeze(2) 
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) 
        node_scores = node_scores.unsqueeze(2) 

        if return_attention_weights:
            gnn_output, (edge_idx, edge_weight) = self.gnn(gnn_input, adj, node_type_ids, node_scores)
        else:
            gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:,0]   

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) 

        mask = mask | (node_type_ids == 3) 
        mask[mask.all(1), 0] = 0  

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)


        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)) 
        
        logits = self.fc(concat) 
        logits = self.activateOut(logits)
        if return_attention_weights:
            return logits, pool_attn, (edge_idx, edge_weight)
        else:
            return logits, pool_attn


class LM_QAGNN(nn.Module):
    def __init__(self, args, pre_dim, model_name, k, n_ntype, n_etype,
                n_concept, concept_dim, concept_in_dim, n_attention_head,
                fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                pretrained_concept_emb=None, freeze_ent_emb=True,
                init_range=0.0, encoder_config={}, hita_config={}):
        super().__init__()
        
        self.encoder_PreTrain = TextEncoder(model_name, **encoder_config)
        self.encoder_HITA = TransformerTime(**hita_config)

        
        
        self.decoder = QAGNN(args, pre_dim, k, n_ntype, n_etype,
                             self.encoder_PreTrain.sent_dim,
                            n_concept, concept_dim, concept_in_dim, n_attention_head,
                            fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                            pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                            init_range=init_range)
        self.use_gram_emb = True

    
    def forward(self, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True):
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        
        edge_index_orig, edge_type_orig = inputs[-2:]  
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs 
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) 

        if return_hita_attention:
            vecs_hita, visit_att, self_att = self.encoder_HITA(diagnosis_codes, seq_time_step,
                                      mask_mult, mask_final, mask_code,
                                      lengths,seq_time_step2, return_hita_attention) 
        else:
            vecs_hita = self.encoder_HITA(diagnosis_codes, seq_time_step,
                                                     mask_mult, mask_final, mask_code,
                                                     lengths, seq_time_step2, return_hita_attention)
        
        
        sent_vec = vecs_hita 
        
        

        if return_attention_weights:
            logits, attn, (edge_idx, edge_weight) = self.decoder(sent_vec.to(node_type_ids.device),
                                    vecs_hita.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output)
        else:
            logits, attn = self.decoder(sent_vec.to(node_type_ids.device),
                                                             vecs_hita.to(node_type_ids.device),
                                                             concept_ids,
                                                             node_type_ids, node_scores, adj_lengths, adj,
                                                             emb_data=None, cache_output=cache_output)
        

        if not detail:
            if return_attention_weights:
                return logits, attn, (edge_idx, edge_weight), visit_att, self_att
            else:
                return logits, attn
        else:
            if return_attention_weights:
                return logits, attn, concept_ids.view(bs, nc, -1), \
                       node_type_ids.view(bs, nc, -1), edge_index_orig, \
                       edge_type_orig, (edge_idx, edge_weight)
            else:
                return logits, attn, concept_ids.view(bs, nc, -1), \
                       node_type_ids.view(bs, nc, -1), edge_index_orig, \
                       edge_type_orig

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        
        
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) 
        edge_type = torch.cat(edge_type_init, dim=0) 
        return edge_index, edge_type



class LM_QAGNN_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                dev_statement_path, dev_adj_path,
                test_statement_path, test_adj_path,
                batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=20,
                is_inhouse=False, inhouse_train_qids_path=None,
                subsample=1.0, use_cache=True):
        super().__init__() 
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device 
        self.is_inhouse = is_inhouse 

        
        
        model_type = MODEL_NAME_TO_CLASS[model_name] 
        self.train_qids, self.train_HF_labels, self.train_Diag_labels, \
            self.train_diagnosis_codes, self.train_seq_time_step, self.train_mask_mult, \
            self.train_mask_final, self.train_mask_code, self.train_lengths, self.train_seq_time_step2, *self.train_encoder_data= \
            load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_HF_labels, self.dev_Diag_labels, \
            self.dev_diagnosis_codes, self.dev_seq_time_step, self.dev_mask_mult, \
            self.dev_mask_final, self.dev_mask_code, self.dev_lengths, self.dev_seq_time_step2, *self.dev_encoder_data= \
            load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        
        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice:', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)

        if test_statement_path is not None:
            self.test_qids, self.test_HF_labels, self.test_Diag_labels, \
            self.test_diagnosis_codes, self.test_seq_time_step, self.test_mask_mult, \
            self.test_mask_final, self.test_mask_code, self.test_lengths, self.test_seq_time_step2, *self.test_encoder_data = \
                load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)
            
        
        
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            self.train_qids = self.train_qids[:n_train]
            self.train_HF_labels = self.train_HF_labels[:n_train]
            self.train_Diag_labels = self.train_Diag_labels[:n_train]
            self.train_diagnosis_codes = self.train_diagnosis_codes[:n_train]
            self.train_seq_time_step = self.train_seq_time_step[:n_train]
            self.train_seq_time_step2 = self.train_seq_time_step2[:n_train]
            self.train_mask_mult = self.train_mask_mult[:n_train]
            self.train_mask_final = self.train_mask_final[:n_train]
            self.train_mask_code = self.train_mask_code[:n_train]
            self.train_lengths = self.train_lengths[:n_train]
            self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
            self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
            self.train_adj_data = self.train_adj_data[:n_train]
                
            assert self.train_size() == n_train
    
    
    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    
    def dev_size(self):
        return len(self.dev_qids)

    
    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   train_indexes,
                                                   self.train_qids,
                                                   self.train_HF_labels,
                                                   self.train_Diag_labels,
                                                   self.train_diagnosis_codes,
                                                   self.train_seq_time_step,
                                                   self.train_mask_mult,
                                                   self.train_mask_final,
                                                   self.train_mask_code,
                                                   self.train_lengths,
                                                   self.train_seq_time_step2,
                                                   tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data,
                                                   adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval',
                                                   self.device0, self.device1,
                                                   self.eval_batch_size,
                                                   torch.arange(len(self.dev_qids)),
                                                   self.dev_qids,
                                                   self.dev_HF_labels,
                                                   self.dev_Diag_labels,
                                                   self.dev_diagnosis_codes,
                                                   self.dev_seq_time_step,
                                                   self.dev_mask_mult,
                                                   self.dev_mask_final,
                                                   self.dev_mask_code,
                                                   self.dev_lengths,
                                                   self.dev_seq_time_step2,
                                                   tensors0=self.dev_encoder_data,
                                                   tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1,
                                                       self.eval_batch_size,
                                                       torch.arange(len(self.test_qids)),
                                                       self.test_qids,
                                                       self.test_HF_labels,
                                                       self.test_Diag_labels,
                                                       self.test_diagnosis_codes,
                                                       self.test_seq_time_step,
                                                       self.test_mask_mult,
                                                       self.test_mask_final,
                                                       self.test_mask_code,
                                                       self.test_lengths,
                                                       self.test_seq_time_step2,
                                                       tensors0=self.test_encoder_data,
                                                       tensors1=self.test_decoder_data,
                                                       adj_data=self.test_adj_data)









from torch.autograd import Variable
def make_one_hot(labels, C):
    
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter



class GATConvE(MessagePassing):
    
    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=True):

        edge_vec = make_one_hot(edge_type, self.n_etype +1) 
        self_edge_vec = torch.zeros(x.size(0), self.n_etype +1).to(edge_vec.device)
        self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]] 
        tail_type = node_type[edge_index[1]] 
        head_vec = make_one_hot(head_type, self.n_ntype) 
        tail_vec = make_one_hot(tail_type, self.n_ntype) 
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) 
        self_head_vec = make_one_hot(node_type, self.n_ntype) 
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) 

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) 
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) 
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) 
        
        
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings) 
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr): 
        
        
        
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key   = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) 
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) 
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) 


        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2) 
        src_node_index = edge_index[0] 
        alpha = softmax(scores, src_node_index) 
        self._alpha = alpha

        
        E = edge_index.size(1)            
        N = int(src_node_index.max()) + 1 
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index] 
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1) 

        out = msg * alpha.view(-1, self.head_count, 1) 
        return out.view(-1, self.head_count * self.dim_per_head)  
