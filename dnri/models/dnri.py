import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from . import model_utils
from .model_utils import RefNRIMLP, encode_onehot
import math


class DNRI(nn.Module):
    def __init__(self, params):
        super(DNRI, self).__init__()
        # Model Params
        self.num_vars = params['num_vars']
        self.encoder = DNRI_Encoder(params)
        decoder_type = params.get('decoder_type', None)
        if decoder_type == 'ref_mlp':
            self.decoder = DNRI_MLP_Decoder(params)
        else:
            self.decoder = DNRI_Decoder(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        
        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 10.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.add_uniform_prior = params.get('add_uniform_prior')
        if self.add_uniform_prior:
            if params.get('no_edge_prior') is not None:
                prior = np.zeros(self.num_edge_types)
                prior.fill((1 - params['no_edge_prior'])/(self.num_edge_types - 1))
                prior[0] = params['no_edge_prior']
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior
                print("USING NO EDGE PRIOR: ",self.log_prior)
            else:
                print("USING UNIFORM PRIOR")
                prior = np.zeros(self.num_edge_types)
                prior.fill(1.0/self.num_edge_types)
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior
                
                
            # this should be a new param called 'no_term_prior', and is a quick fix for now
            if params.get('no_edge_prior') is not None:
                prior = np.zeros(2)
                prior.fill(0.7)#1 - params['no_term_prior']))
                prior[0] = 0.3#params['no_term_prior']
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_term_prior = log_prior
                print("USING NO EDGE PRIOR: ",self.log_prior)
            else:
                print("USING UNIFORM PRIOR")
                prior = np.zeros(2)
                prior.fill(0.5)
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_term_prior = log_prior

    def single_step_forward(self, inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges)
        return predictions, decoder_hidden, edges
    
    def single_step_critic_val(self, inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        critic_vals = self.decoder.return_critic(inputs, decoder_hidden, edges)
        return critic_vals
    
    def calc_contrastive_loss(self, inputs1, decoder_hidden1, edge_logits1, hard_sample1, inputs2, decoder_hidden2, edge_logits2, hard_sample2):
        old_shape1 = edge_logits1.shape
        edges1 = model_utils.gumbel_softmax(
            edge_logits1.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample1).view(old_shape1)
        mu1, sig1 = self.decoder.return_mu_sigma(inputs1, decoder_hidden1, edges1)
        
        old_shape2 = edge_logits2.shape
        edges2 = model_utils.gumbel_softmax(
            edge_logits2.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample2).view(old_shape2)
        mu2, sig2 = self.decoder.return_mu_sigma(inputs2, decoder_hidden2, edges2)
        
        kld_loss = (0.5 * torch.sum(2*torch.log(sig2/sig1) + (sig1/sig2)**2 + ((mu1 - mu2)/sig2)**2 - 1, dim = -1) ).mean(dim = -1)
        
        return kld_loss

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False, disc=None):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_priors = []
        
        all_critic_vals = []
        
        # intervention variables
        all_interventions = []
        interv_decoder_hidden = self.decoder.get_initial_hidden(inputs)
        
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, posterior_logits, _, prior_term, posterior_term = self.encoder(inputs[:, :-1])
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        for step in range(num_time_steps-1):
            if (teacher_forcing) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            if not use_prior_logits:
                current_p_logits = posterior_logits[:, step]
            else:
                current_p_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            critic_vals = self.single_step_critic_val(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            
            # conducting an intervention
            intervened_indices = torch.randint(current_p_logits.shape[1], (current_p_logits.shape[0],)).cuda()
            logit_to_flip = torch.randint(current_p_logits.shape[2], (current_p_logits.shape[0],)).cuda()
            intervened_p_logits = current_p_logits.clone()
            # This temp_storage will be used for the reparameterization trick
            temp_storage = current_p_logits.detach().clone()
            
            for i in range(intervened_indices.shape[0]):
                # if we want to flip [a,b], we need to add [b-a, a-b], i.e. [b-a, a-b] + [a,b] = [b,a]
                max_ind = temp_storage[i, intervened_indices[i]].argmax()
                
                while (max_ind == logit_to_flip[i]).item():
                    # pick another logit to flip
                    logit_to_flip[i] = np.random.randint(temp_storage.shape[2])
                        
                sum_to_flip = temp_storage[i, intervened_indices[i], max_ind] - temp_storage[i, intervened_indices[i], logit_to_flip[i]]
                intervened_p_logits[i, intervened_indices[i], logit_to_flip[i]] += sum_to_flip
                intervened_p_logits[i, intervened_indices[i], max_ind] -= sum_to_flip
            
            # interv_predictions, interv_decoder_hidden, _ = self.single_step_forward(current_inputs, interv_decoder_hidden, intervened_p_logits, hard_sample)
            # all_interventions.append(interv_predictions)
            kld_loss = self.calc_contrastive_loss(current_inputs, interv_decoder_hidden, intervened_p_logits, hard_sample,
                                                  current_inputs, decoder_hidden, current_p_logits, hard_sample)
            all_interventions.append(kld_loss)
            
            all_predictions.append(predictions)
            all_edges.append(edges)
            all_critic_vals.append(critic_vals)
        all_predictions = torch.stack(all_predictions, dim=1)
        all_critic_vals = torch.stack(all_critic_vals, dim=1)
        # interventions
        all_interventions = torch.stack(all_interventions, dim=-1)
        
        final_critic_vals = self.single_step_critic_val(current_inputs, decoder_hidden, current_p_logits, hard_sample)
        
        if disc is not None:
            x1_x2_pairs = torch.cat([all_predictions[:, :-1, :, :], all_predictions[:, 1:, :, :]], dim=-1)
            discrim_pred = disc(x1_x2_pairs)
            discrim_prob = F.softmax(discrim_pred, dim=-1)
            disc_entropy = discrim_prob * torch.log(discrim_prob + 1e-16)
            disc_entropy = disc_entropy.sum(-1)
            
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        
        target_values_critic = loss_nll.detach().clone()
        
        gamma = 0.99
        for i in reversed(range(len(loss_nll[1]))):
            if i == len(loss_nll[1]) - 1:
                loss_nll[:,i] += final_critic_vals.mean(dim=-1)*gamma - all_critic_vals[:,i].mean(dim=-1)
                target_values_critic[:,i] += final_critic_vals.mean(dim=-1)*gamma
                
            else:
                loss_nll[:,i] += loss_nll[:,i+1]*gamma - all_critic_vals[:,i].mean(dim=-1)
                target_values_critic[:,i] += target_values_critic[:,i+1]*gamma
        
        loss_nll = loss_nll.mean(dim=-1)
        loss_critic = 0.5 * self.nll(all_critic_vals.clone().mean(dim=-1), target_values_critic).mean(dim=-1)

        reinforce_loss = loss_nll.detach().view(loss_nll.shape[0], loss_nll.shape[1], 1, 1) * torch.log(all_predictions.clone())
        loss_nll_reinf = reinforce_loss.mean(dim=[1,2,3])
        
        loss_nll = loss_nll.mean(dim=-1)
        all_interventions = all_interventions.mean(dim=-1)
        
        #intervention loss
        # intervention_loss_nll = self.nll(all_interventions, all_predictions).mean(dim=-1)
        intervention_loss_nll = all_interventions
        
        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)
        prob_term = F.softmax(posterior_term, dim=-1)
        loss_kl_term = self.kl_categorical_learned(prob_term, prior_term)
        if self.add_uniform_prior:
            loss_kl = 0.33*loss_kl + 0.33*self.kl_categorical_avg(prob) + 0.33*self.kl_categorical_avg_term(prob_term)
        
        # intervention cap bounds the contrastive loss
        intervention_cap = 1.
        loss = loss_nll_reinf + loss_critic + self.kl_coef*(loss_kl + loss_kl_term) + torch.max(intervention_cap - intervention_loss_nll, torch.zeros(intervention_loss_nll.shape).cuda())
        
        if disc is not None:
            loss = loss.mean() - self.kl_coef*disc_entropy.mean()
        else:
            loss = loss.mean()

        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        all_predictions = []
        all_edges = []
        prior_logits, _, prior_hidden, _, _ = self.encoder(inputs[:, :-1])
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
            if return_everything:
                all_edges.append(edges)
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden, current_edge_logits)
            predictions, decoder_hidden, edges = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
            all_predictions.append(predictions)
            all_edges.append(edges)
        
        predictions = torch.stack(all_predictions, dim=1)
        if return_edges:
            edges = torch.stack(all_edges, dim=1)
            return predictions, edges
        else:
            return predictions

    def copy_states(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            current_state = (state[0].clone(), state[1].clone())
        else:
            current_state = state.clone()
        return current_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        print("INPUT SHAPE: ",inputs.shape)
        prior_logits, _, prior_hidden,_,_ = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        all_timestep_edges = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            current_batch_edges = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step] 
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden, current_edge_logits)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior = self.encoder.copy_states(prior_hidden)
                tmp_decoder = self.copy_states(decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
                if return_edges:
                    current_batch_edges.append(current_edge_logits.cpu())
            batch_prior_hidden = self.encoder.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            if return_edges:
                current_batch_edges = torch.cat(current_batch_edges, 0)
                current_timestep_edges = [current_batch_edges]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden, current_batch_edge_logits)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
                if return_edges:
                    current_timestep_edges.append(current_batch_edge_logits.cpu())
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
            if return_edges:
                all_timestep_edges.append(torch.stack(current_timestep_edges, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        if return_edges:
            edge_result = torch.cat(all_timestep_edges, dim=0)
            return result.unsqueeze(0), edge_result.unsqueeze(0)
        else:
            return result.unsqueeze(0)

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const)#.mean(dim=-1)#.view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))


    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_avg(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds*(torch.log(avg_preds+eps) - self.log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)


    def kl_categorical_avg_term(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds*(torch.log(avg_preds+eps) - self.log_term_prior)
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class DNRI_Encoder(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(DNRI_Encoder, self).__init__()
        num_vars = params['num_vars']
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden']
        rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        out_hidden_size = 2*rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.encoder_fc_out = nn.Sequential(*layers)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)


        # adding the termination model here
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.term_encoder_fc_out = nn.Linear(out_hidden_size, 1)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, 1))
            self.term_encoder_fc_out = nn.Sequential(*layers)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.term_prior_fc_out = nn.Linear(rnn_hidden_size, 1)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, 1))
            self.term_prior_fc_out = nn.Sequential(*layers)
        
        
        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?


    def copy_states(self, prior_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        return current_prior_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            result = (result0, result1)
        else:
            result = torch.cat(hidden, dim=0)
        return result



    def forward(self, inputs):
        if self.training or not self.save_eval_memory:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            x = inputs.transpose(2, 1).contiguous()
            # New shape: [num_sims, num_atoms, num_timesteps, num_dims]
            x = self.mlp1(x)  # 2-layer ELU net per node
            x = self.node2edge(x)
            x = self.mlp2(x)
            x_skip = x
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        
            
            # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
            # RNN aggregation
            old_shape = x.shape
            x = x.contiguous().view(-1, old_shape[2], old_shape[3])
            forward_x, prior_state = self.forward_rnn(x)
            timesteps = old_shape[2]
            reverse_x = x.flip(1)
            reverse_x, _ = self.reverse_rnn(reverse_x)
            reverse_x = reverse_x.flip(1)
            
            #x: [batch*num_edges, num_timesteps, hidden_size]
            prior_result = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            combined_x = torch.cat([forward_x, reverse_x], dim=-1)
            encoder_result = self.encoder_fc_out(combined_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            
            # adding term models here
            term_prior_result = self.term_prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps, 1).transpose(1,2).contiguous()
            term_encoder_result = self.term_encoder_fc_out(combined_x).view(old_shape[0], old_shape[1], timesteps, 1).transpose(1,2).contiguous()
            
            term_prior_prob = torch.sigmoid(term_prior_result)
            term_encoder_prob = torch.sigmoid(term_encoder_result)
            
            # these two lines are the wrong implementation. We need a rolling carry for the termination (which the for loop achieves)
            # prior_result[:,1:] = prior_result[:,1:]*term_prior_prob[:,1:] + prior_result[:,:-1]*(1.0 - term_prior_prob[:,1:])
            # encoder_result[:,1:] = encoder_result[:,1:]*term_encoder_prob[:,1:] + encoder_result[:,:-1]*(1.0 - term_encoder_prob[:,1:])
            
            z_prev_prior = torch.clone(prior_result)
            z_prev_encoder = torch.clone(encoder_result)
            
            for i in range(1, timesteps):
                term_vals = term_prior_prob[:,i]
                prior_result[:,i] = z_prev_prior[:,i]*term_vals + z_prev_prior[:,i-1]*(1.0 - term_vals)
                
                term_enc_vals = term_encoder_prob[:,i]
                encoder_result[:,i] = z_prev_encoder[:,i]*term_enc_vals + z_prev_encoder[:,i-1]*(1.0 - term_enc_vals)
            # 
            # prior_result = prior_result*term_prior_result + z_prev_prior*(1.0-term_prior_result)
            # encoder_result = encoder_result*term_encoder_result + z_prev_encoder*(1.0 - term_encoder_result)

            return prior_result, encoder_result, prior_state, term_prior_result, term_encoder_result
        else:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            all_x = []
            all_forward_x = []
            all_prior_result = []
            prior_state = None
            for timestep in range(num_timesteps):
                x = inputs[:, timestep]
                #x = inputs.transpose(2, 1).contiguous()
                x = self.mlp1(x)  # 2-layer ELU net per node
                x = self.node2edge(x)
                x = self.mlp2(x)
                x_skip = x
                x = self.edge2node(x)
                x = self.mlp3(x)
                x = self.node2edge(x)
                x = torch.cat((x, x_skip), dim=-1)  # Skip connection
                x = self.mlp4(x)
            
                
                # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
                # RNN aggregation
                old_shape = x.shape
                x = x.contiguous().view(-1, 1, old_shape[-1])
                forward_x, prior_state = self.forward_rnn(x, prior_state)
                all_x.append(x.cpu())
                all_forward_x.append(forward_x.cpu())
                all_prior_result.append(self.prior_fc_out(forward_x).view(old_shape[0], 1, old_shape[1], self.num_edges).cpu())
            reverse_state = None
            all_encoder_result = []
            for timestep in range(num_timesteps-1, -1, -1):
                x = all_x[timestep].cuda()
                reverse_x, reverse_state = self.reverse_rnn(x, reverse_state)
                forward_x = all_forward_x[timestep].cuda()
                
                #x: [batch*num_edges, num_timesteps, hidden_size]
                combined_x = torch.cat([forward_x, reverse_x], dim=-1)
                all_encoder_result.append(self.encoder_fc_out(combined_x).view(inputs.size(0), 1, -1, self.num_edges))
            prior_result = torch.cat(all_prior_result, dim=1).cuda(non_blocking=True)
            encoder_result = torch.cat(all_encoder_result, dim=1).cuda(non_blocking=True)
            return prior_result, encoder_result, prior_state

    def single_step_forward(self, inputs, prior_state, z_prev_prior):
        # Inputs is shape [batch, num_vars, input_size]
        x = self.mlp1(inputs)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        x = self.mlp4(x)

        old_shape = x.shape
        x  = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]),
                       prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]))

        x, prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        prior_state = (prior_state[0].view(old_prior_shape), prior_state[1].view(old_prior_shape))
        
        # adding the term model here
        term_prior_result = self.term_prior_fc_out(x).view(old_shape[0], old_shape[1], 1)
        term_prior_prob = torch.sigmoid(term_prior_result)
        prior_result = prior_result*term_prior_prob + z_prev_prior*(1.0-term_prior_prob)
        
        return prior_result, prior_state

    
class DNRI_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_Decoder, self).__init__()
        self.num_vars = num_vars =  params['num_vars']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, num_edges, num_edge_types]
        if self.training:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0.
        
        # node2edge
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch, num_edges, 2*msg_out]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape, device=inputs.device)
        
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i+1]
            all_msgs += msg/norm

        # This step sums all of the messages per node
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1) # Average

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        pred = self.out_fc3(pred)

        pred = inputs + pred

        return pred, hidden


class DNRI_MLP_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_MLP_Decoder, self).__init__()
        num_vars = params['num_vars']
        edge_types = params['num_edge_types']
        n_hid = params['decoder_hidden']
        msg_hid = params['decoder_hidden']
        msg_out = msg_hid #TODO: make this a param
        skip_first = params['skip_first']
        n_in_node = params['input_size']

        do_prob = params['decoder_dropout']
        in_size = n_in_node
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_size, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        out_size = n_in_node
        self.out_fc1 = nn.Linear(in_size + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.mu_layer = nn.Linear(n_hid, out_size)
        self.log_std_layer = nn.Linear(n_hid, out_size)
        
        self.critic_layer = nn.Linear(n_hid, 1)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob
        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return None
    
    def return_critic(self, inputs, hidden, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
            std_gating = 1.
        else:
            p = 0
            std_gating = 0.

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=p)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=p)
        
        critic_val = self.critic_layer(pred)

        return critic_val
    
    def return_mu_sigma(self, inputs, hidden, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
            std_gating = 1.
        else:
            p = 0
            std_gating = 0.

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=p)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=p)
        
        mu = self.mu_layer(pred)
        log_std = self.log_std_layer(pred)
        log_std = torch.clamp(log_std, -3, 1)
        std = torch.exp(log_std)

        return mu, std

    def forward(self, inputs, hidden, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
            std_gating = 1.
        else:
            p = 0
            std_gating = 0.

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=p)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=p)
        
        mu = self.mu_layer(pred)
        log_std = self.log_std_layer(pred)
        log_std = torch.clamp(log_std, -3, 1)
        std = torch.exp(log_std)
        
        pred = mu + std_gating * torch.normal(torch.zeros(mu.shape), torch.ones(std.shape)).cuda() * std

        # Predict position/velocity difference
        return inputs + pred, None

class DNRI_Disc(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(DNRI_Disc, self).__init__()
        num_vars = params['num_vars']
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden']
        
        # The input size is double the dimension of x, as we
        # are taking [ x_t, x_t+1 ] pairs
        inp_size = 2*params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)


        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?


    def copy_states(self, prior_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        return current_prior_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            result = (result0, result1)
        else:
            result = torch.cat(hidden, dim=0)
        return result



    def forward(self, inputs):
        if self.training or not self.save_eval_memory:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            x = inputs.transpose(2, 1).contiguous()
            # New shape: [num_sims, num_atoms, num_timesteps, num_dims]
            x = self.mlp1(x)  # 2-layer ELU net per node
            x = self.node2edge(x)
            x = self.mlp2(x)
            x_skip = x
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        
            
            # At this point, x should be [batch, num_vars, num_timesteps, hidden_size]
            old_shape = x.shape
            
            #x: [batch*num_vars, num_timesteps, hidden_size]
            prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], old_shape[2], self.num_edges).transpose(1,2).contiguous()
            return prior_result

    def single_step_forward(self, inputs, prior_state):
        # Inputs is shape [batch, num_vars, input_size]
        x = self.mlp1(inputs)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        x = self.mlp4(x)

        old_shape = x.shape

        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        return prior_result
