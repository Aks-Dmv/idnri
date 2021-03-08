import torch
from torch import nn
import torch.nn.functional as F
from . import model_utils
import numpy as np

class RecurrentBaseline_DynamicVars(nn.Module):
    def __init__(self, params):
        super(RecurrentBaseline_DynamicVars, self).__init__()
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

    def normalize_inputs(self, inputs, masks):
        return inputs
    
    def calculate_loss(self, inputs, node_masks, node_inds, graph_info, is_train=False, teacher_forcing=True, return_logits=False, use_prior_logits=False, normalized_inputs=None):
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
            current_node_masks = node_masks[:, step]
            node_inds = current_node_masks.nonzero()[:, -1]
            num_edges = len(node_inds)*(len(node_inds)-1)
            current_graph_info = graph_info[0][step]
            predictions, hidden = self.single_step_forward(current_inputs, current_node_masks, current_graph_info, hidden)
            all_predictions.append(predictions)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        target_masks = ((node_masks[:, :-1] == 1)*(node_masks[:, 1:] == 1)).float()
        loss_nll = self.nll(all_predictions, target, target_masks)
        loss_kl = torch.zeros_like(loss_nll)
        loss = loss_nll.mean()
        if return_logits:
            return loss, loss_nll, loss_kl, None, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, masks, node_inds, graph_info, burn_in_masks):
        '''
        Here, we assume the following:
        * inputs contains all of the gt inputs, including for the time steps we're predicting
        * masks keeps track of the variables that are being tracked
        * burn_in_masks is set to 1 whenever we're supposed to feed in that variable's state
          for a given time step
        '''
        total_timesteps = inputs.size(1)
        hidden = self.get_initial_hidden(inputs)
        predictions = inputs[:, 0]
        preds = []
        for step in range(total_timesteps-1):
            current_masks = masks[:, step]
            current_burn_in_masks = burn_in_masks[:, step].unsqueeze(-1).type(inputs.dtype)
            current_inps = inputs[:, step]
            current_node_inds = node_inds[0][step]
            current_graph_info = graph_info[0][step]
            decoder_inp = current_burn_in_masks*current_inps + (1-current_burn_in_masks)*predictions
            predictions, hidden = self.single_step_forward(decoder_inp, current_masks, current_graph_info, hidden)
            preds.append(predictions)
        return torch.stack(preds, dim=1)
        
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

    def nll(self, preds, target, masks):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target, masks)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target, masks)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target, masks)

    def nll_gaussian(self, preds, target, masks, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))*masks.unsqueeze(-1)
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            raise NotImplementedError()
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const*masks).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1)+1e-8)
        else:
            raise NotImplementedError()


    def nll_crossent(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def nll_poisson(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.PoissonNLLLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class FullyConnectedBaseline_DynamicVars(RecurrentBaseline_DynamicVars):
    def __init__(self, params):
        super(FullyConnectedBaseline_DynamicVars, self).__init__(params)
        n_hid = params['decoder_hidden']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']
        input_size = params['input_size']

        self.msg_fc1 = nn.Linear(2*n_hid, n_hid)
        self.msg_fc2 = nn.Linear(n_hid, n_hid)
        self.msg_out_shape = n_hid

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

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)


    def single_step_forward(self, inputs, node_masks, graph_info, hidden):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, num_edges, num_edge_types]
        if self.training:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0.
        
        max_num_vars = inputs.size(1)
        node_inds = node_masks.nonzero()[:, -1]
        current_hidden = hidden[:, node_inds]
        current_inputs = inputs[:, node_inds]
        num_vars = current_hidden.size(1)
        if num_vars > 1:
            send_edges, recv_edges, edge2node_inds = graph_info
            send_edges, recv_edges, edge2node_inds = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True), edge2node_inds.cuda(non_blocking=True)

            receivers = current_hidden[:, recv_edges]
            senders = current_hidden[:, send_edges]

            # pre_msg: [batch, num_edges, 2*msg_out]
            pre_msg = torch.cat([receivers, senders], dim=-1)

            msg = torch.tanh(self.msg_fc1(pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.msg_fc2(msg))
            all_msgs = msg

            incoming = all_msgs[:, edge2node_inds[:, 0], :].clone()
            for i in range(1, edge2node_inds.size(1)):
                incoming += all_msgs[:, edge2node_inds[:, i], :]
            agg_msgs = incoming/(num_vars-1)
        elif num_vars == 0:
            pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
            return pred_all, hidden
        else:
            agg_msgs = torch.zeros(current_inputs.size(0), num_vars, self.msg_out_shape, device=inputs.device)


        # GRU-style gated aggregation
        inp_r = self.input_r(current_inputs).view(current_inputs.size(0), num_vars, -1)
        inp_i = self.input_i(current_inputs).view(current_inputs.size(0), num_vars, -1)
        inp_n = self.input_n(current_inputs).view(current_inputs.size(0), num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        current_hidden = (1 - i)*n + i*current_hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(current_hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        pred = self.out_fc3(pred)

        pred = current_inputs + pred
        hidden = hidden.clone()
        hidden[:, node_inds] = current_hidden
        pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
        pred_all[0, node_inds] = pred

        return pred_all, hidden