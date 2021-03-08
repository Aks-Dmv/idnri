import torch
from torch import nn
import torch.nn.functional as F
from . import model_utils
import numpy as np

from .model_utils import encode_onehot

class RecurrentBaseline(nn.Module):
    def __init__(self, params):
        super(RecurrentBaseline, self).__init__()
        self.num_vars = params['num_vars']
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.anneal_teacher_forcing = params.get('anneal_teacher_forcing', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.kl_coef = 0
        self.steps = 0
    
    def single_step_forward(self, inputs, hidden):
        raise NotImplementedError

    def get_initial_hidden(self, inputs):
        raise NotImplementedError

    def normalize_inputs(self, inputs):
        return inputs
    
    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_logits=False, use_prior_logits=False, normalized_inputs=None):
        hidden = self.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_predictions = []
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            predictions, hidden = self.single_step_forward(current_inputs, hidden)
            all_predictions.append(predictions)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        loss_kl = torch.zeros_like(loss_nll)
        loss = loss_nll.mean()
        if return_logits:
            return loss, loss_nll, loss_kl, None, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        hidden = self.get_initial_hidden(inputs)
        all_predictions = []
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            predictions, hidden = self.single_step_forward(current_inputs, hidden)
            if return_everything:
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            predictions, hidden = self.single_step_forward(predictions, hidden)
            all_predictions.append(predictions)
        
        predictions = torch.stack(all_predictions, dim=1)
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

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size):
        hidden = self.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            predictions, hidden = self.single_step_forward(current_inputs, hidden)
        all_timestep_preds = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step]
                predictions, hidden = self.single_step_forward(predictions, hidden)
                current_batch_preds.append(predictions)
                tmp_decoder = self.copy_states(hidden)
                states.append(tmp_decoder)
            batch_hidden = self.merge_hidden(states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            for step in range(prediction_steps - 1):
                current_batch_preds, batch_hidden = self.single_step_forward(current_batch_preds, batch_hidden)
                current_timestep_preds.append(current_batch_preds)
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
        results = torch.cat(all_timestep_preds, dim=0)
        return results.unsqueeze(0)

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
            return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)
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

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class SingleRNNBaseline(RecurrentBaseline):
    def __init__(self, params):
        super(SingleRNNBaseline, self).__init__(params)
        self.n_hid = n_hid = params['decoder_hidden']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']
        input_size = params['input_size']
        self.num_vars = num_vars = params['num_vars']
        self.rnn_type = params['decoder_rnn_type']
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(input_size, n_hid)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRUCell(input_size, n_hid)
        self.out = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, out_size),
        )

    def get_initial_hidden(self, inputs):
        return None
        '''
        if self.rnn_type == 'lstm':
            raise NotImplementedError
            return torch.zeros(inputs.size(0)*inputs.size(2), self.n_hid, device=inputs.device)
        else:
            return torch.zeros(inputs.size(0)*inputs.size(2), self.n_hid, device=inputs.device)
        '''

    def single_step_forward(self, inputs, hidden):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        tmp_inp = inputs.reshape(-1, inputs.size(-1))
        hidden = self.rnn(tmp_inp)
        if self.rnn_type == 'lstm':
            tmp = hidden[0].view(inputs.size(0), inputs.size(1), -1)
        else:
            tmp = hidden.view(inputs.size(0), inputs.size(1), -1)
        outputs = inputs + self.out(tmp)
        return outputs, hidden

class JointRNNBaseline(RecurrentBaseline):
    def __init__(self, params):
        super(JointRNNBaseline, self).__init__(params)
        self.n_hid = n_hid = params['decoder_hidden']
        do_prob = params['decoder_dropout']
        self.num_vars = num_vars = params['num_vars']
        out_size = input_size = params['input_size']*num_vars
        self.rnn_type = params['decoder_rnn_type']
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(input_size, n_hid)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRUCell(input_size, n_hid)
        self.out = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, out_size),
        )

    def get_initial_hidden(self, inputs):
        return None
        '''
        if self.rnn_type == 'lstm':
            raise NotImplementedError
            return torch.zeros(inputs.size(0)*inputs.size(2), self.n_hid, device=inputs.device)
        else:
            return torch.zeros(inputs.size(0)*inputs.size(2), self.n_hid, device=inputs.device)
        '''

    def single_step_forward(self, inputs, hidden):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        tmp_inp = inputs.view(inputs.size(0), -1)
        hidden = self.rnn(tmp_inp)
        if self.rnn_type == 'lstm':
            tmp = hidden[0]
        else:
            tmp = hidden
        outputs = inputs + self.out(tmp).view(inputs.size(0), inputs.size(1), -1)
        return outputs, hidden
 
        


class FullyConnectedBaseline(RecurrentBaseline):
    def __init__(self, params):
        super(FullyConnectedBaseline, self).__init__(params)
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']
        input_size = params['input_size']
        self.num_vars = num_vars =  params['num_vars']

        self.msg_fc1 = nn.Linear(2*n_hid, n_hid)
        self.msg_fc2 = nn.Linear(n_hid, n_hid)
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


    def single_step_forward(self, inputs, hidden):
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

        msg = torch.tanh(self.msg_fc1(pre_msg))
        msg = F.dropout(msg, p=dropout_prob)
        msg = torch.tanh(self.msg_fc2(msg))
        all_msgs = msg

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