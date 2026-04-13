"""
Created on Sep 15th 18:05:53 2020
Author: Qizhong Zhang
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from data_prepare import mycollatefunc

matplotlib.use("Agg")

# Set the random seeds
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Enable tf32 for float32 matmuls (3x speedup on H100 with negligible precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cudnn benchmark for consistent input sizes
torch.backends.cudnn.benchmark = True


"""
Embedding Modules
"""


class ABS_TIM_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.d_model = param.d_model
        self.device = param.device
        self.fourier = param.fourier
        # Pre-compute the frequency vector once (moved out of forward)
        a = torch.div(torch.arange(0.0, self.d_model), 2, rounding_mode="floor") * 2 / self.d_model
        if not self.fourier:
            self.register_buffer("freq", (2 * np.pi / 1440) * a)
        else:
            self.register_buffer("freq", 1e-4**a)

    def forward(self, x):
        b = torch.matmul(x.unsqueeze(-1), self.freq.unsqueeze(0))
        c = torch.zeros_like(b)
        c[:, :, 0::2] = b[:, :, 0::2].sin()
        c[:, :, 1::2] = b[:, :, 1::2].cos()
        return c


class TIM_DIFF_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.tim_emb_type = param.tim_emb_type
        self.tim_size = param.tim_size
        self.tim_emb_size = param.tim_emb_size
        if self.tim_emb_type == "Linear":
            self.emb_tim = nn.Linear(1, self.tim_emb_size, bias=False)
        else:
            self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size, padding_idx=0)

    def forward(self, x):
        if self.tim_emb_type == "Linear":
            return self.emb_tim(torch.log(x.unsqueeze(-1) + 1e-10))
        return self.emb_tim(x.long())


class LOC_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.loc_size = param.loc_size + 1
        self.loc_emb_size = param.loc_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size, padding_idx=0)

    def forward(self, x):
        return self.emb_loc(x)


class USR_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.usr_size = param.usr_size + 1
        self.usr_emb_size = param.usr_emb_size
        self.emb_usr = nn.Embedding(self.usr_size, self.usr_emb_size, padding_idx=0)
        self.device = param.device

        # Pre-compute user lookup table as a torch tensor on device
        USERLIST = np.append(-1, param.USERLIST) + 1
        # Build a direct mapping: user_id -> index (avoids np.where in forward)
        max_user_id = int(USERLIST.max()) + 1
        lookup = torch.zeros(max_user_id, dtype=torch.long)
        for idx, uid in enumerate(USERLIST):
            lookup[int(uid)] = idx
        self.register_buffer("user_lookup", lookup)

    def forward(self, x):
        # x contains user IDs; use the pre-computed lookup table (pure torch, no numpy)
        x_shifted = (x + 1).long().clamp(0, self.user_lookup.shape[0] - 1)
        usr2id = self.user_lookup[x_shifted]
        return self.emb_usr(usr2id)


class POI_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        POI = np.concatenate((np.zeros((1, param.POI.shape[1])), param.POI), axis=0)
        if param.cdfpoi:
            CDF = [np.append(0, (np.cumsum(np.bincount(poi.astype(int))) / poi.shape[0]))[:-1] for poi in POI.T]
            POI = np.array([[CDF[i][int(j)] for j in POI[:, i]] for i in range(POI.shape[1])]).T
        else:
            POI = np.log((POI + 1))
        # Store as a buffer tensor on device instead of numpy array
        self.register_buffer("POI", torch.tensor(POI, dtype=torch.float32))

    def forward(self, x):
        # Pure torch indexing, no numpy conversion in forward
        return self.POI[x.long()]


"""
Initialization functions
"""


def initialize_rnn(model, type="LSTM"):
    for name, param in model.named_parameters():
        if "weight_ih" in name:
            torch.nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            if type == "LSTM":
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.orthogonal_(param)
        elif "bias" in name:
            torch.nn.init.constant_(param, 0.0)


def initialize_linear_layer(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight, a=0.01)


"""
Encoder
"""


class ENCODER(nn.Module):

    def __init__(self, param):
        super(ENCODER, self).__init__()

        self.device = param.device
        self.d_model = param.d_model if not param.pos_emb_ban else 0
        self.poi_size = param.poi_size if not param.poi_emb_ban else 0

        self.encoder_rnn_input_size = (
            param.loc_emb_size + param.tim_emb_size + param.usr_emb_size + self.d_model + self.poi_size
        )
        self.encoder_rnn_hidden_size = param.encoder_rnn_hidden_size
        self.rnn_type = param.rnn_type
        self.rnn_layers = param.rnn_layers
        self.rnn_bidirectional = param.rnn_bidirectional

        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.encoder_rnn = rnn_cls(
            self.encoder_rnn_input_size,
            self.encoder_rnn_hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.rnn_bidirectional,
        )

        self.layernorm = param.layernorm
        self.layer_norm = nn.LayerNorm(self.encoder_rnn_hidden_size * (1 + self.rnn_bidirectional))

        self.z_hidden_size_mean = param.z_hidden_size_mean
        self.z_hidden_size_std = param.z_hidden_size_std
        self.latent_size = param.latent_size

        self.mean_l1 = nn.Linear(self.encoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.z_hidden_size_mean)
        self.mean_l2 = nn.Linear(self.z_hidden_size_mean, self.latent_size)
        self.std_l1 = nn.Linear(self.encoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.z_hidden_size_std)
        self.std_l2 = nn.Linear(self.z_hidden_size_std, self.latent_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                initialize_linear_layer(m)
            if isinstance(m, (nn.LSTM, nn.GRU)):
                initialize_rnn(m, self.rnn_type)

    def forward(self, x_emb, lengths=None):
        lengths = lengths if lengths is not None else [x_emb.shape[1]] * x_emb.shape[0]
        packed_input = pack_padded_sequence(input=x_emb, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.encoder_rnn(packed_input)
        hidden0, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x_emb.shape[1])

        hidden = self.layer_norm(hidden0) if self.layernorm else hidden0

        mean = self.mean_l2(F.leaky_relu(self.mean_l1(hidden)))
        std = self.std_l2(F.leaky_relu(self.std_l1(hidden)))

        return mean, std


"""
Decoder
"""


class DECODER(nn.Module):

    def __init__(self, param):
        super(DECODER, self).__init__()

        self.device = param.device
        self.d_model = param.d_model if not param.feedback_ban else 0
        self.poi_ban = param.poi_ban

        self.decoder_rnn_input_size = param.latent_size + param.usr_emb_size + self.d_model
        self.decoder_rnn_hidden_size = param.decoder_rnn_hidden_size
        self.rnn_type = param.rnn_type
        self.rnn_layers = param.rnn_layers
        self.rnn_bidirectional = param.rnn_bidirectional
        self.dual_rnn = param.dual_rnn

        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.decoder_rnn = rnn_cls(
            self.decoder_rnn_input_size,
            self.decoder_rnn_hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.rnn_bidirectional,
        )
        if self.dual_rnn:
            self.decoder_rnn1 = rnn_cls(
                self.decoder_rnn_input_size,
                self.decoder_rnn_hidden_size,
                num_layers=self.rnn_layers,
                batch_first=True,
                bidirectional=self.rnn_bidirectional,
            )

        self.layernorm = param.layernorm
        self.layer_norm = nn.LayerNorm(self.decoder_rnn_hidden_size * (1 + self.rnn_bidirectional))
        self.dropout = nn.Dropout(param.dropout)

        self.loc_hidden_size1 = param.loc_hidden_size1
        self.loc_hidden_size2 = param.loc_hidden_size2
        self.loc_size = param.loc_size + 1
        self.poi_size = param.poi_size

        self.loc_l1 = nn.Linear(self.decoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.loc_hidden_size1)
        self.loc_l2 = nn.Linear(self.loc_hidden_size1, self.loc_hidden_size2)
        self.loc_l3 = nn.Linear(self.loc_hidden_size2, self.loc_size)

        if self.poi_size and not self.poi_ban:
            self.loc_l4 = nn.Linear(self.loc_hidden_size2, self.poi_size)
            self.loc_l5 = nn.Linear(self.poi_size, 1, bias=None)

            POI = np.concatenate((np.zeros((1, param.POI.shape[1])), param.POI), axis=0)
            if param.cdfpoi:
                CDF = [np.append(0, (np.cumsum(np.bincount(poi.astype(int))) / poi.shape[0]))[:-1] for poi in POI.T]
                POI = np.array([[CDF[i][int(j)] for j in POI[:, i]] for i in range(POI.shape[1])]).T
            else:
                POI = np.log((POI + 1))

            # Store POI as buffer (on device, no numpy in forward)
            self.register_buffer("POI_tensor", torch.tensor(POI, dtype=torch.float32))

            if param.poi_weight_dynamic:
                self.poi_weight = nn.Parameter(torch.tensor(-np.log(1 / param.poi_weight - 1), dtype=torch.float32))
            else:
                self.register_buffer("poi_weight", torch.tensor(-np.log(1 / param.poi_weight - 1), dtype=torch.float32))

        self.tim_hidden_size1 = param.tim_hidden_size1
        self.tim_hidden_size2 = param.tim_hidden_size2

        self.tim_l1 = nn.Linear(self.decoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.tim_hidden_size1)
        self.tim_l2 = nn.Linear(self.tim_hidden_size1, self.tim_hidden_size2)
        self.tim_l3 = nn.Linear(self.tim_hidden_size2, 1)
        self.time_initial = param.time_initial
        # Pre-compute log(time_initial) as a buffer
        self.register_buffer("log_time_initial", torch.tensor(np.log(self.time_initial), dtype=torch.float32))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                initialize_linear_layer(m)
            if isinstance(m, (nn.LSTM, nn.GRU)):
                initialize_rnn(m, self.rnn_type)

    def forward(self, z, lengths=None):
        lengths = lengths if lengths is not None else [z.shape[1]] * z.shape[0]
        packed_input = pack_padded_sequence(input=z, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.decoder_rnn(packed_input)
        hidden0, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=z.shape[1])
        hidden = self.layer_norm(hidden0) if self.layernorm else hidden0
        if self.dual_rnn:
            packed_output1, _ = self.decoder_rnn1(packed_input)
            hiddenT, _ = pad_packed_sequence(packed_output1, batch_first=True, total_length=z.shape[1])
            hidden1 = self.layer_norm(hiddenT) if self.layernorm else hiddenT

        # Location decoder
        lout2 = F.leaky_relu(self.loc_l2(F.leaky_relu(self.loc_l1(self.dropout(hidden)))))
        if self.poi_size and not self.poi_ban:
            lout_1 = F.log_softmax(self.loc_l3(lout2), dim=2)
            POI = self.POI_tensor
            lout_2 = F.log_softmax(self.loc_l5(F.leaky_relu(self.loc_l4(lout2)).unsqueeze(-2) * POI).squeeze(-1), dim=2)
            lout = torch.logaddexp(
                torch.log(1 - torch.sigmoid(self.poi_weight)) + lout_1, F.logsigmoid(self.poi_weight) + lout_2
            )
        else:
            lout = F.log_softmax(self.loc_l3(lout2), dim=2)

        # Time decoder
        tout = self.tim_l3(F.leaky_relu(self.tim_l2(F.leaky_relu(self.tim_l1(hidden))))).squeeze(-1)
        if self.dual_rnn:
            tout = self.tim_l3(F.leaky_relu(self.tim_l2(F.leaky_relu(self.tim_l1(hidden1))))).squeeze(-1)

        return lout, tout - self.log_time_initial


"""
VAE MODEL
"""


class VAE(nn.Module):

    def __init__(self, param):

        super().__init__()

        self.tim_size = param.tim_size
        self.poi_size = param.poi_size
        self.loc_size = param.loc_size + 1
        self.latent_size = param.latent_size
        self.pos_emb_ban = param.pos_emb_ban
        self.poi_emb_ban = param.poi_emb_ban
        self.feedback_ban = param.feedback_ban

        # Embedding
        self.emb_loc = LOC_EMB(param)
        self.emb_tim = TIM_DIFF_EMB(param)
        self.emb_usr = USR_EMB(param)
        self.emb_pos = ABS_TIM_EMB(param) if not self.pos_emb_ban else None
        self.emb_poi = POI_EMB(param) if ((self.poi_emb_ban == False) and (self.poi_size > 0)) else None

        self.emb_pos0 = ABS_TIM_EMB(param) if not self.feedback_ban else None
        self.emb_usr0 = USR_EMB(param)
        for para_0, para in zip(self.emb_usr0.parameters(), self.emb_usr.parameters()):
            para_0.data.copy_(para.data)
            para_0.requires_grad = False

        self.encoder = ENCODER(param)
        self.decoder = DECODER(param)

        # Generation params
        self.USERLIST = param.USERLIST
        self.infer_maxlast = param.infer_maxlast
        self.first_sample = param.first_sample
        self.ntrajs = param.ntrajs

        # KL-annealing
        self.max_beta = param.max_beta
        self.cycle = param.cycle
        # Learning
        self.learning_rate = param.learning_rate
        self.L2 = param.L2
        self.step_size = param.step_size
        self.gamma = param.gamma
        self.epoches = param.epoches
        self.batchsize = param.batchsize

        # Tuning
        self.tune = param.tune
        self.tuned = 0

        # Pre-compute user_indicator and loc_weights as torch tensors
        self.user_indicator = np.ones((param.USERLIST.shape[0] + 1, param.loc_size + 1))
        self.loc_weights = np.ones((param.USERLIST.shape[0] + 1, param.loc_size + 1))
        self.initial_prob = np.ones(self.tim_size // 10)
        self.save_path = param.save_path
        self.device = param.device
        self.loc_initial = param.loc_initial

        # Build user lookup as a GPU tensor (avoid numpy in hot path)
        USERLIST_shifted = np.append(-1, self.USERLIST)
        max_uid = int(USERLIST_shifted.max()) + 2
        _lookup = torch.zeros(max_uid, dtype=torch.long)
        for idx, uid in enumerate(USERLIST_shifted):
            _lookup[int(uid) + 1] = idx  # +1 because ids can be -1
        self.register_buffer("_user_index_lookup", _lookup)

        # Cached tensors for locprob_filter (set after location_constraints)
        self._user_indicator_tensor = None
        self._user_indicator_binary = None  # for loss LL_L0

        # Unbalanced learning
        self.tim_only = param.tim_only
        if self.tim_only:
            timdecoder = ["tim_l1", "tim_l2", "tim_l3"]
            for name, para in self.named_parameters():
                modulename, layername = name.split(".")[0], name.split(".")[1]
                if (layername not in timdecoder) and (modulename != "emb_usr0"):
                    para.requires_grad = False

        self.loc_only = param.loc_only
        if self.loc_only:
            locdecoder = ["loc_l1", "loc_l2", "loc_l3", "loc_l4", "loc_l5"]
            for name, para in self.named_parameters():
                modulename, layername = name.split(".")[0], name.split(".")[1]
                if (layername not in locdecoder) and (modulename != "emb_usr0"):
                    para.requires_grad = False

    def _cache_indicator_tensor(self):
        """Cache user_indicator as GPU tensors to avoid numpy->torch in forward."""
        self._user_indicator_tensor = torch.tensor(self.user_indicator, dtype=torch.float32, device=self.device)
        # Binary version: 1 where indicator == 1, else 0 (used in loss LL_L0)
        self._user_indicator_binary = (self._user_indicator_tensor == 1).float()

    def forward(self, inseq):

        with torch.no_grad():
            self._embedding_update()

        loc_emb = self.emb_loc(inseq["loc"] + 1)
        tim_emb = self.emb_tim(inseq["tim"])
        usr_emb = self.emb_usr(inseq["usr"] + 1)
        pos_emb = self.emb_pos(inseq["pos"]) if not self.pos_emb_ban else torch.empty(0, device=self.device)
        poi_emb = (
            self.emb_poi(inseq["loc"] + 1)
            if ((self.poi_emb_ban == False) and (self.poi_size > 0))
            else torch.empty(0, device=self.device)
        )
        x_emb = torch.cat((loc_emb, poi_emb, tim_emb, usr_emb, pos_emb), -1)

        mean, logstd = self.encoder(x_emb, inseq["lengths"])

        z = torch.randn_like(logstd)
        z = z * torch.exp(logstd) + mean

        usr_emb0 = self.emb_usr0(inseq["usr"] + 1)
        pos_emb0 = self.emb_pos0(inseq["pos"]) if not self.feedback_ban else torch.empty(0, device=self.device)
        z = torch.cat((z, usr_emb0, pos_emb0), -1)

        lout, tout = self.decoder(z, inseq["lengths"])
        lout = self.locprob_filter(lout, inseq["usr"])

        return mean, logstd, lout, tout

    def locprob_filter(self, lout, usr, training=True):
        usr_shifted = (usr + 1).long().clamp(0, self._user_index_lookup.shape[0] - 1)
        user_encoding = self._user_index_lookup[usr_shifted]  # GPU tensor lookup

        if self._user_indicator_tensor is not None:
            weights = self._user_indicator_tensor[user_encoding]
        else:
            # Fallback before caching (shouldn't happen during training)
            weights = torch.tensor(self.user_indicator, dtype=torch.float32, device=self.device)[user_encoding]

        l0 = lout - torch.log(weights) if training else lout.masked_fill(weights != 1, float("-inf"))
        l1 = l0 - torch.logsumexp(l0, dim=2, keepdim=True)
        return l1

    @torch.no_grad()
    def _embedding_update(self):
        for param_0, param in zip(self.emb_usr0.parameters(), self.emb_usr.parameters()):
            param_0.data = param.data

    def loss(self, mean, logstd, lout, tout, inseq, weight=0.5, beta=1):
        # Efficient mask: use arange broadcast instead of pad_sequence with list comp
        lengths_t = torch.as_tensor(inseq["lengths"], dtype=torch.long, device=self.device)
        max_len = lout.shape[1]
        mask = (torch.arange(max_len, device=self.device).unsqueeze(0) < lengths_t.unsqueeze(1)).float()
        percentage = torch.sum(mask) / (mask.shape[0] * mask.shape[1])

        KL = (
            torch.mean(0.5 * torch.sum(torch.exp(2 * logstd) - 1 - 2 * logstd + mean.pow(2), dim=-1) * mask)
            / percentage
        )

        LL_T = -torch.mean((tout - torch.exp(tout) * inseq["tim"]) * mask) / percentage
        LL_T0 = (
            -torch.mean((torch.log1p(-torch.exp(-torch.exp(tout))) - torch.exp(tout) * inseq["tim"]) * mask)
            / percentage
        )

        LL_L = torch.mean(nn.NLLLoss(reduction="none")(lout.swapaxes(1, 2), inseq["loc"] + 1) * mask) / percentage

        # GPU-resident indicator computation (no numpy)
        usr_shifted = (inseq["usr"] + 1).long().clamp(0, self._user_index_lookup.shape[0] - 1)
        user_encoding = self._user_index_lookup[usr_shifted]
        if self._user_indicator_binary is not None:
            indicator = self._user_indicator_binary[user_encoding]
        else:
            indicator = (
                torch.tensor(self.user_indicator, dtype=torch.float32, device=self.device)[user_encoding] == 1
            ).float()

        LL_L0 = nn.NLLLoss(reduction="none")((lout * indicator).swapaxes(1, 2), inseq["loc"] + 1) * mask
        LL_L0 = torch.mean(LL_L0) / (
            torch.sum(LL_L0 != 0) / torch.cumprod(torch.tensor(LL_L0.shape, device=self.device), 0)[-1]
        )

        LOSS = beta * KL + (1 - weight) * LL_T + weight * LL_L

        return KL, LL_L, LL_T, LOSS, LL_L0, LL_T0

    def sample(self, lout, tout, last_loc, ntrajs=7):
        Lambda = torch.exp(tout[:, -1]).unsqueeze(1).cpu().detach().numpy()

        def truncated_exponential_samples(size, lower, upper, rate):
            uniform_samples = np.random.rand(*size)
            rate0 = np.where(rate < 1e-10, 1e-10, rate)
            truncated_samples = -np.log(1 - uniform_samples * (1 - np.exp(-rate0 * (upper - lower)))) / rate0 + lower
            return truncated_samples

        t = np.squeeze(truncated_exponential_samples((ntrajs, 1), 1, self.infer_maxlast, Lambda))

        prob = torch.exp(lout[:, -1, :]).squeeze(1).cpu().detach()
        for id, loc in enumerate(last_loc):
            prob[id, 0] = 0
            prob[id, loc] = 0
        prob = prob / prob.sum(dim=1, keepdim=True)
        l = torch.multinomial(prob, 1).squeeze(1).numpy()

        return l, t

    @torch.no_grad()
    def inference(self, user, ntrajs=7):
        with torch.no_grad():
            if self.feedback_ban:
                z = torch.randn(ntrajs, 144, self.latent_size, device=self.device)
                usr = (user + 1) * torch.ones((ntrajs, 144), dtype=torch.long, device=self.device)
                last_location = np.zeros(ntrajs).astype(int)

                X = {"loc": [], "tim": [], "sta": []}
                lout, tout = self.decoder(torch.cat((z, self.emb_usr0(usr)), -1))
                lout = self.locprob_filter(lout, usr - 1)

                for i in range(1, 145):
                    l, t = self.sample(lout[:, :i, :], tout[:, :i], last_location)
                    X["loc"].append(l)
                    X["sta"].append(t)
                    last_location = l.astype(int)
                    if np.min(np.sum(np.array(X["sta"]), axis=0)) >= self.infer_maxlast:
                        break

                output = {}
                for i in range(ntrajs):
                    id = np.where(np.cumsum(np.array(X["sta"])[:, i]) >= self.infer_maxlast)[0][0]
                    output[i] = {
                        "loc": np.array(X["loc"])[1 : (1 + id), i] - 1,
                        "tim": np.cumsum(np.array(X["sta"])[:id, i]),
                        "sta": np.array(X["sta"])[1 : (1 + id), i],
                    }
                return output

            z = torch.randn(ntrajs, 1000, self.latent_size, device=self.device)
            usr = (user + 1) * torch.ones((ntrajs, 1000), dtype=torch.long, device=self.device)

            if self.first_sample == "New":
                t = np.random.choice(np.arange(self.tim_size // 10), size=self.ntrajs, p=self.initial_prob) * 10 + 5
                last_location = np.zeros(ntrajs).astype(int)
            else:
                usr_emb0 = self.emb_usr0(usr[:, 0].unsqueeze(1))
                pos_emb0 = self.emb_pos0(torch.ones((ntrajs, 1), dtype=torch.float32, device=self.device) * 10)
                lout1, tout1 = self.decoder(torch.cat((z[:, 0, :].unsqueeze(1), usr_emb0, pos_emb0), -1))
                lout1 = self.locprob_filter(lout1, usr[:, 0].unsqueeze(1) - 1)
                l, t = self.sample(lout1, tout1, last_location, ntrajs=ntrajs)
                last_location = l.astype(int)

            X = {"loc": [], "tim": [t], "sta": []}
            for i in range(1, 1000):
                time = torch.tensor(np.array(X["tim"]).T, dtype=torch.float32, device=self.device)
                usr_emb0 = self.emb_usr0(usr[:, :i])
                pos_emb0 = self.emb_pos0(time)
                louti, touti = self.decoder(torch.cat((z[:, :i, :], usr_emb0, pos_emb0), -1))
                louti = self.locprob_filter(louti, usr[:, :i] - 1)

                l, t = self.sample(louti, touti, last_location)
                X["tim"].append(np.array(X["tim"])[-1] + t)
                X["loc"].append(l)
                X["sta"].append(t)
                last_location = np.array(X["loc"])[-1].astype(int)

                if np.min(np.array(X["tim"])[-1]) >= self.infer_maxlast:
                    break

            output = {}
            for i in range(ntrajs):
                id = np.where(np.array(X["tim"])[:, i] >= self.infer_maxlast)[0][0]
                output[i] = {
                    "loc": np.array(X["loc"])[:id, i] - 1,
                    "tim": np.array(X["tim"])[:id, i],
                    "sta": np.array(X["sta"])[:id, i],
                }
            return output

    @torch.no_grad()
    def inference_batch(self, users, ntrajs=7, max_steps=1000):
        n_users = len(users)
        B = n_users * ntrajs

        users_rep = np.repeat(users, ntrajs)
        usr_single = torch.tensor(users_rep + 1, dtype=torch.long, device=self.device).unsqueeze(1)
        usr_emb0 = self.emb_usr0(usr_single)

        t0 = np.random.choice(np.arange(self.tim_size // 10), size=B, p=self.initial_prob) * 10 + 5
        cum_time = torch.tensor(t0, dtype=torch.float32, device=self.device)
        last_location = torch.zeros(B, dtype=torch.long, device=self.device)

        X_loc = []
        X_tim = [cum_time.cpu().numpy().copy()]
        X_sta = []

        rnn = self.decoder.decoder_rnn
        h = None

        step_bar = tqdm(range(1, max_steps), desc=f"Decoding {B} trajectories")
        for i in step_bar:
            z_step = torch.randn(B, 1, self.latent_size, device=self.device)

            if self.feedback_ban:
                dec_input = torch.cat((z_step, usr_emb0), -1)
            else:
                pos_emb0 = self.emb_pos0(cum_time.unsqueeze(1))
                dec_input = torch.cat((z_step, usr_emb0, pos_emb0), -1)

            dec_out, h = rnn(dec_input, h)
            hidden = self.decoder.layer_norm(dec_out) if self.decoder.layernorm else dec_out

            lout2 = F.leaky_relu(self.decoder.loc_l2(F.leaky_relu(self.decoder.loc_l1(hidden))))
            if self.decoder.poi_size and not self.decoder.poi_ban:
                POI = self.decoder.POI_tensor
                lout_1 = F.log_softmax(self.decoder.loc_l3(lout2), dim=2)
                lout_2 = F.log_softmax(
                    self.decoder.loc_l5(F.leaky_relu(self.decoder.loc_l4(lout2)).unsqueeze(-2) * POI).squeeze(-1), dim=2
                )
                lout = torch.logaddexp(
                    torch.log(1 - torch.sigmoid(self.decoder.poi_weight)) + lout_1,
                    F.logsigmoid(self.decoder.poi_weight) + lout_2,
                )
            else:
                lout = F.log_softmax(self.decoder.loc_l3(lout2), dim=2)

            tout = self.decoder.tim_l3(
                F.leaky_relu(self.decoder.tim_l2(F.leaky_relu(self.decoder.tim_l1(hidden))))
            ).squeeze(-1)
            tout = tout - self.decoder.log_time_initial

            lout = self.locprob_filter(lout, usr_single - 1, training=False)

            rate = torch.exp(tout[:, -1]).clamp(min=1e-10)
            u = torch.rand(B, device=self.device)
            t_gpu = -torch.log(1 - u * (1 - torch.exp(-rate * (self.infer_maxlast - 1)))) / rate + 1

            prob = torch.exp(lout[:, -1, :]).squeeze(1)
            prob[:, 0] = 0
            prob.scatter_(1, last_location.unsqueeze(1), 0)
            prob = prob / prob.sum(dim=1, keepdim=True)
            l_gpu = torch.multinomial(prob, 1).squeeze(1)

            cum_time = cum_time + t_gpu
            last_location = l_gpu

            l_np = l_gpu.cpu().numpy()
            t_np = t_gpu.cpu().numpy()
            X_loc.append(l_np)
            X_tim.append(cum_time.cpu().numpy().copy())
            X_sta.append(t_np)

            done = (cum_time >= self.infer_maxlast).sum().item()
            step_bar.set_postfix(done=f"{done}/{B}")
            if done == B:
                break

        step_bar.close()

        locs_arr = np.array(X_loc)
        tims_arr = np.array(X_tim)
        stas_arr = np.array(X_sta)

        results = []
        for u_idx in range(n_users):
            user_output = {}
            for k in range(ntrajs):
                j = u_idx * ntrajs + k
                traj_tims = tims_arr[:, j]
                time_hits = np.where(traj_tims >= self.infer_maxlast)[0]
                end_idx = time_hits[0] if len(time_hits) > 0 else locs_arr.shape[0]

                user_output[k] = {
                    "loc": locs_arr[:end_idx, j] - 1,
                    "tim": tims_arr[:end_idx, j],
                    "sta": stas_arr[:end_idx, j],
                }
            results.append(user_output)

        return results

    @torch.no_grad()
    def inference_od_batch(self, users, origins, destinations, ntrajs=7, max_steps=1000):
        """Batched OD-conditioned generation for multiple OD pairs at once.

        All computation stays on GPU until final output assembly.

        Args:
            users: np.array of user IDs, shape (n_pairs,)
            origins: np.array of origin location IDs (0-indexed), shape (n_pairs,)
            destinations: np.array of destination location IDs (0-indexed), shape (n_pairs,)
            ntrajs: number of trajectories per OD pair
            max_steps: maximum decoding steps

        Returns:
            list of dicts, one per OD pair, each mapping traj_index -> {loc, tim, sta}
        """
        n_pairs = len(users)
        B = n_pairs * ntrajs  # total parallel trajectories

        # Repeat each OD pair ntrajs times
        users_rep = np.repeat(users, ntrajs)
        origins_rep = np.repeat(origins, ntrajs) + 1  # shifted
        dests_rep = np.repeat(destinations, ntrajs) + 1  # shifted

        # Pre-compute constant tensors on GPU
        usr_single = torch.tensor(users_rep + 1, dtype=torch.long, device=self.device).unsqueeze(1)  # (B, 1)
        usr_emb0 = self.emb_usr0(usr_single)  # constant across steps, compute once
        dests_t = torch.tensor(dests_rep, dtype=torch.long, device=self.device)

        # State tensors on GPU
        cum_time = torch.ones(B, device=self.device) * 5.0
        last_location = torch.tensor(origins_rep, dtype=torch.long, device=self.device)
        reached = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Store results as lists of numpy arrays (single CPU transfer per step)
        X_loc = [origins_rep.copy()]
        X_tim = [cum_time.cpu().numpy().copy()]
        X_sta = []

        rnn = self.decoder.decoder_rnn
        h = None

        for i in range(1, max_steps):
            z_step = torch.randn(B, 1, self.latent_size, device=self.device)

            if self.feedback_ban:
                dec_input = torch.cat((z_step, usr_emb0), -1)
            else:
                pos_emb0 = self.emb_pos0(cum_time.unsqueeze(1))
                dec_input = torch.cat((z_step, usr_emb0, pos_emb0), -1)

            # Single-step RNN with carried hidden state
            dec_out, h = rnn(dec_input, h)
            hidden = self.decoder.layer_norm(dec_out) if self.decoder.layernorm else dec_out

            # Location head
            lout2 = F.leaky_relu(self.decoder.loc_l2(F.leaky_relu(self.decoder.loc_l1(hidden))))
            if self.decoder.poi_size and not self.decoder.poi_ban:
                POI = self.decoder.POI_tensor
                lout_1 = F.log_softmax(self.decoder.loc_l3(lout2), dim=2)
                lout_2 = F.log_softmax(
                    self.decoder.loc_l5(F.leaky_relu(self.decoder.loc_l4(lout2)).unsqueeze(-2) * POI).squeeze(-1), dim=2
                )
                lout = torch.logaddexp(
                    torch.log(1 - torch.sigmoid(self.decoder.poi_weight)) + lout_1,
                    F.logsigmoid(self.decoder.poi_weight) + lout_2,
                )
            else:
                lout = F.log_softmax(self.decoder.loc_l3(lout2), dim=2)

            # Time head
            tout = self.decoder.tim_l3(
                F.leaky_relu(self.decoder.tim_l2(F.leaky_relu(self.decoder.tim_l1(hidden))))
            ).squeeze(-1)
            tout = tout - self.decoder.log_time_initial

            lout = self.locprob_filter(lout, usr_single - 1, training=False)

            # --- Time sampling (GPU) ---
            rate = torch.exp(tout[:, -1]).clamp(min=1e-10)
            u = torch.rand(B, device=self.device)
            t_gpu = -torch.log(1 - u * (1 - torch.exp(-rate * (self.infer_maxlast - 1)))) / rate + 1

            # --- Location sampling (GPU, vectorized masking) ---
            prob = torch.exp(lout[:, -1, :]).squeeze(1)
            prob[:, 0] = 0
            prob.scatter_(1, last_location.unsqueeze(1), 0)
            prob = prob / prob.sum(dim=1, keepdim=True)
            l_gpu = torch.multinomial(prob, 1).squeeze(1)

            # Update state
            cum_time = cum_time + t_gpu
            last_location = l_gpu

            # Single CPU transfer per step for result storage
            l_np = l_gpu.cpu().numpy()
            t_np = t_gpu.cpu().numpy()
            X_loc.append(l_np)
            X_tim.append(cum_time.cpu().numpy().copy())
            X_sta.append(t_np)

            # Termination check (GPU)
            reached |= l_gpu == dests_t
            if torch.all(reached | (cum_time >= self.infer_maxlast)):
                break

        # --- Build per-OD-pair outputs ---
        locs_arr = np.array(X_loc)  # (steps, B)
        tims_arr = np.array(X_tim)  # (steps, B)
        stas_arr = np.array(X_sta)  # (steps-1, B)

        results = []
        for p in range(n_pairs):
            pair_output = {}
            for k in range(ntrajs):
                j = p * ntrajs + k
                traj_locs = locs_arr[:, j]
                traj_tims = tims_arr[:, j]
                dest_shifted = dests_rep[j]

                dest_hits = np.where(traj_locs[1:] == dest_shifted)[0]
                time_hits = np.where(traj_tims >= self.infer_maxlast)[0]

                if len(dest_hits) > 0:
                    end_idx = dest_hits[0] + 2
                elif len(time_hits) > 0:
                    end_idx = time_hits[0]
                else:
                    end_idx = len(traj_locs)

                pair_output[k] = {
                    "loc": traj_locs[1:end_idx] - 1,
                    "tim": traj_tims[1:end_idx],
                    "sta": stas_arr[: end_idx - 1, j] if end_idx - 1 <= stas_arr.shape[0] else stas_arr[:, j],
                }
            results.append(pair_output)

        return results

    def load(self, cp):
        self.load_state_dict(torch.load(cp, map_location=self.device))
        print("Load model from %s" % cp)

    def save(self, cp):
        torch.save(self.state_dict(), cp)
        print("Model saved as %s" % cp)

    @torch.no_grad()
    def test_data_prepare(self, data, load_checkpoint=None, batch_size=128):
        self.eval()
        if load_checkpoint is not None:
            self.load(load_checkpoint)

        test_users = list(data.REFORM["test"].keys())
        total = len(test_users)
        all_users = np.array(test_users)

        output_sequence = {}
        gen_bar = tqdm(range(0, total, batch_size), desc="Unconditional generation")

        for start in gen_bar:
            end = min(start + batch_size, total)
            batch_results = self.inference_batch(all_users[start:end], ntrajs=self.ntrajs)
            for i, user in enumerate(all_users[start:end]):
                output_sequence[user] = batch_results[i]
            gen_bar.set_postfix(done=f"{end}/{total}")

        data.GENDATA.append(output_sequence)
        np.save(self.save_path + "data/generated_" + str(self.tuned) + ".npy", output_sequence)
        self.tuned += 1

    @torch.no_grad()
    def test_data_prepare_od(self, data, batch_size=128):
        """Batched OD-conditioned generation from test data.

        Flattens all test trajectories, batches OD pairs together, and runs
        inference_od_batch for GPU-efficient parallel generation.
        """
        self.eval()
        test_data = data.REFORM["test"]

        # Flatten all (user, traj_idx, traj) into arrays
        all_users, all_origins, all_dests, all_keys, all_labels = [], [], [], [], []
        for user, trajs in test_data.items():
            for traj_idx, traj in trajs.items():
                all_users.append(user)
                all_origins.append(int(traj["loc"][0]))
                all_dests.append(int(traj["loc"][-1]))
                all_keys.append((user, traj_idx))
                all_labels.append(traj)

        all_users = np.array(all_users)
        all_origins = np.array(all_origins)
        all_dests = np.array(all_dests)
        total = len(all_users)

        output_sequence = {}
        label_sequence = {}
        gen_bar = tqdm(range(0, total, batch_size), desc="OD-conditioned generation")

        for start in gen_bar:
            end = min(start + batch_size, total)
            batch_results = self.inference_od_batch(
                all_users[start:end], all_origins[start:end], all_dests[start:end], ntrajs=self.ntrajs
            )
            for i, (user, traj_idx) in enumerate(all_keys[start:end]):
                output_sequence.setdefault(user, {})[traj_idx] = batch_results[i]
                label_sequence.setdefault(user, {})[traj_idx] = all_labels[start + i]
            gen_bar.set_postfix(done=f"{end}/{total}")

        np.save(self.save_path + "data/generated_od.npy", output_sequence)
        np.save(self.save_path + "data/labels_od.npy", label_sequence)
        print(f"Saved generated trajectories and labels to {self.save_path}data/")

    def location_constraints(self, data):
        user_indicator = np.ones((self.USERLIST.shape[0] + 1, self.loc_size))
        weights = np.ones((self.USERLIST.shape[0] + 1, self.loc_size))
        for userid, data_user in data.items():
            count = np.bincount(
                np.concatenate([traj["loc"] for traj in data_user.values()]),
                minlength=self.loc_size,
            )[: self.loc_size]
            row = np.where(self.USERLIST == userid)[0][0] + 1
            weights[row] = count
            user_indicator[(row, count == 0)] = self.loc_initial
        weights[weights > 0] = np.log(weights[weights > 0]) + 1
        return user_indicator, weights

    def inference_initial(self, data):
        timestamps = np.concatenate(
            [[traj["tim"][0] % 1440 for traj in trajs.values()] for trajs in data.dataset.REFORM["train"].values()]
        )
        freqs = np.histogram(timestamps, bins=self.tim_size // 10, range=(0, self.tim_size))[0]
        return freqs / freqs.sum()

    def Train(self, trainset, validset):

        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.L2,
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        self.user_indicator, self.loc_weights = self.location_constraints(trainset.dataset.REFORM["train"])
        self._cache_indicator_tensor()
        self.initial_prob = self.inference_initial(trainset)

        compiled_forward = torch.compile(self.forward, mode="default", dynamic=True)

        amp_dtype = torch.bfloat16

        # KL-annealing
        def cyclical_KL_annealing(step, cycle, R=0.5, M=1):
            step0 = step % cycle
            M = M * step0 / (cycle * R) if step0 < cycle * R else M
            return M, step + 1

        loss_record, valid_record, step, weight = {}, {}, 0, 0.5

        for epoch in range(1, self.epoches + 1):
            loss_record[epoch] = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": [], "weight": [], "LL_L0": [], "LL_T0": []}

            mydataloader = DataLoader(
                trainset,
                batch_size=self.batchsize,
                shuffle=True,
                num_workers=4,
                collate_fn=mycollatefunc,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
            train_bar = tqdm(enumerate(mydataloader), total=len(mydataloader))

            for idx, bat in train_bar:
                loc = bat["loc"].long().to(self.device, non_blocking=True)
                tim = bat["sta"].float().to(self.device, non_blocking=True)
                usr = bat["usr"].long().to(self.device, non_blocking=True)
                pos = bat["tim"].float().to(self.device, non_blocking=True)
                inseq = {"usr": usr, "loc": loc, "tim": tim, "pos": pos, "lengths": bat["lengths"]}

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    mean, std, lout, tout = compiled_forward(inseq)

                    beta, step = cyclical_KL_annealing(step, self.cycle, M=self.max_beta)
                    if self.tim_only:
                        beta, weight = 0, 0
                    if self.loc_only:
                        beta, weight = 0, 1
                    KL, LL_L, LL_T, LOSS, LL_L0, LL_T0 = self.loss(
                        mean, std, lout, tout, inseq, weight=weight, beta=beta
                    )

                optimizer.zero_grad(set_to_none=True)
                LOSS.backward()
                optimizer.step()

                # Loss record
                loss_record[epoch]["KL"].append(KL.item())
                loss_record[epoch]["LL_L"].append(LL_L.item())
                loss_record[epoch]["LL_T"].append(LL_T.item())
                loss_record[epoch]["LOSS"].append(LOSS.item())
                loss_record[epoch]["LL_L0"].append(LL_L0.item())
                loss_record[epoch]["LL_T0"].append(LL_T0.item())

                train_bar.set_description(
                    "Ep:{} w:{:.3f} KL:{:.4f} L:{:.4f} L0:{:.4f} T:{:.4f} T0:{:.4f} ELBO:{:.4f}".format(
                        epoch,
                        weight,
                        np.mean(loss_record[epoch]["KL"]),
                        np.mean(loss_record[epoch]["LL_L"]),
                        np.mean(loss_record[epoch]["LL_L0"]),
                        np.mean(loss_record[epoch]["LL_T"]),
                        np.mean(loss_record[epoch]["LL_T0"]),
                        np.mean(loss_record[epoch]["LOSS"]),
                    )
                )

            # ---- Validation (using the actual val.data split) ----
            if epoch % 1 == 0:
                self.eval()
                with torch.no_grad():
                    valid_record[epoch] = {
                        "LOSS": [],
                        "KL": [],
                        "LL_T": [],
                        "LL_L": [],
                        "LL_L0": [],
                        "LL_T0": [],
                        "weight": [],
                    }
                    val_loader = DataLoader(
                        validset,
                        batch_size=self.batchsize,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=mycollatefunc,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    valid_bar = tqdm(enumerate(val_loader))
                    for idx, bat in valid_bar:
                        loc = bat["loc"].long().to(self.device, non_blocking=True)
                        tim = bat["sta"].float().to(self.device, non_blocking=True)
                        usr = bat["usr"].long().to(self.device, non_blocking=True)
                        pos = bat["tim"].float().to(self.device, non_blocking=True)
                        inseq = {"usr": usr, "loc": loc, "tim": tim, "pos": pos, "lengths": bat["lengths"]}

                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            mean, std, lout, tout = self.forward(inseq)
                            KL, LL_L, LL_T, LOSS, LL_L0, LL_T0 = self.loss(
                                mean, std, lout, tout, inseq, weight=weight, beta=beta
                            )

                        valid_record[epoch]["KL"].append(KL.item())
                        valid_record[epoch]["LL_L"].append(LL_L.item())
                        valid_record[epoch]["LL_T"].append(LL_T.item())
                        valid_record[epoch]["LOSS"].append(LOSS.item())
                        valid_record[epoch]["LL_L0"].append(LL_L0.item())
                        valid_record[epoch]["LL_T0"].append(LL_T0.item())

                        valid_bar.set_description(
                            "Val Ep:{} KL:{:.4f} L:{:.4f} L0:{:.4f} T:{:.4f} T0:{:.4f} ELBO:{:.4f}".format(
                                epoch,
                                np.mean(valid_record[epoch]["KL"]),
                                np.mean(valid_record[epoch]["LL_L"]),
                                np.mean(valid_record[epoch]["LL_L0"]),
                                np.mean(valid_record[epoch]["LL_T"]),
                                np.mean(valid_record[epoch]["LL_T0"]),
                                np.mean(valid_record[epoch]["LOSS"]),
                            )
                        )

                self.train()

            weight = 1 / (1 + np.var(loss_record[epoch]["LL_L"]) / np.var(loss_record[epoch]["LL_T"]))
            loss_record[epoch]["weight"].append(weight)
            scheduler.step()

            if epoch % 10 == 0:
                if self.tune:
                    self.save(self.save_path + "data/Model" + str(epoch // 10) + ".pth")
                else:
                    self.save(self.save_path + "data/Model.pth")

        self.save(self.save_path + "data/Model.pth")
        return loss_record, valid_record

    def test(self, testset):
        self.eval()
        with torch.no_grad():
            test_record = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": []}
            mydataloader = DataLoader(
                testset,
                batch_size=self.batchsize,
                shuffle=False,
                num_workers=4,
                collate_fn=mycollatefunc,
                pin_memory=True,
            )
            test_bar = tqdm(enumerate(mydataloader))
            for idx, bat in test_bar:
                loc = bat["loc"].long().to(self.device, non_blocking=True)
                tim = bat["sta"].float().to(self.device, non_blocking=True)
                usr = bat["usr"].long().to(self.device, non_blocking=True)
                pos = bat["tim"].float().to(self.device, non_blocking=True)
                inseq = {"usr": usr, "loc": loc, "tim": tim, "pos": pos, "lengths": bat["lengths"]}

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mean, std, lout, tout = self.forward(inseq)
                    KL, LL_L, LL_T, LOSS, LL_L0, LL_T0 = self.loss(mean, std, lout, tout, inseq)

                test_record["KL"].append(KL.item())
                test_record["LL_L"].append(LL_L.item())
                test_record["LL_T"].append(LL_T.item())
                test_record["LOSS"].append(LOSS.item())

                test_bar.set_description(
                    "Test idx:{} KL:{:.4f} L:{:.4f} T:{:.4f} ELBO:{:.4f}".format(
                        idx,
                        np.mean(test_record["KL"]),
                        np.mean(test_record["LL_L"]),
                        np.mean(test_record["LL_T"]),
                        np.mean(test_record["LOSS"]),
                    )
                )

        # Standard unconditional generation
        if self.tune:
            for i in range(1, 1 + self.epoches // 10):
                self.test_data_prepare(testset.dataset, load_checkpoint=self.save_path + "data/Model" + str(i) + ".pth")
        else:
            self.test_data_prepare(testset.dataset)

        # OD-conditioned generation from the test data
        print("Generating OD-conditioned trajectories from test data...")
        self.test_data_prepare_od(testset.dataset)

        return test_record

    def loss_plot(self, loss_record, valid_record, test_record):
        KL = [np.mean(loss_record[epoch]["KL"]) for epoch in loss_record]
        LL_L = [np.mean(loss_record[epoch]["LL_L"]) for epoch in loss_record]
        LL_T = [np.mean(loss_record[epoch]["LL_T"]) for epoch in loss_record]
        LOSS = [np.mean(loss_record[epoch]["LOSS"]) for epoch in loss_record]

        valid_KL = [np.mean(valid_record[epoch]["KL"]) for epoch in valid_record]
        valid_LL_L = [np.mean(valid_record[epoch]["LL_L0"]) for epoch in valid_record]
        valid_LL_T = [np.mean(valid_record[epoch]["LL_T"]) for epoch in valid_record]
        valid_LOSS = [np.mean(valid_record[epoch]["LOSS"]) for epoch in valid_record]

        x = np.array([epoch for epoch in loss_record])
        y = np.array([epoch for epoch in valid_record])

        plt.figure()

        plt.subplot(221)
        (ln1,) = plt.plot(x, KL, color="red", linewidth=2.0, linestyle="-")
        (ln2,) = plt.plot(y, valid_KL, color="blue", linewidth=2.0, linestyle="-")
        plt.title("KL, Test = " + str(np.around(np.mean(test_record["KL"]), decimals=3)))
        plt.xlabel("Epoches")
        plt.ylabel("KLLoss")
        plt.legend(handles=[ln1, ln2], labels=["Train", "Valid"])

        plt.subplot(222)
        (ln1,) = plt.plot(x, LL_L, color="red", linewidth=2.0, linestyle="-")
        (ln2,) = plt.plot(y, valid_LL_L, color="blue", linewidth=2.0, linestyle="-")
        plt.title("LL_L, Test = " + str(np.around(np.mean(test_record["LL_L"]), decimals=3)))
        plt.xlabel("Epoches")
        plt.ylabel("LL_LLoss")
        plt.legend(handles=[ln1, ln2], labels=["Train", "Valid"])

        plt.subplot(223)
        (ln1,) = plt.plot(x, LL_T, color="red", linewidth=2.0, linestyle="-")
        (ln2,) = plt.plot(y, valid_LL_T, color="blue", linewidth=2.0, linestyle="-")
        plt.title("LL_T, Test = " + str(np.around(np.mean(test_record["LL_T"]), decimals=3)))
        plt.xlabel("Epoches")
        plt.ylabel("LL_TLoss")
        plt.legend(handles=[ln1, ln2], labels=["Train", "Valid"])

        plt.subplot(224)
        (ln1,) = plt.plot(x, LOSS, color="red", linewidth=2.0, linestyle="-")
        (ln2,) = plt.plot(y, valid_LOSS, color="blue", linewidth=2.0, linestyle="-")
        plt.title("ELBO, Test = " + str(np.around(np.mean(test_record["LOSS"]), decimals=3)))
        plt.xlabel("Epoches")
        plt.ylabel("ELBOLoss")
        plt.legend(handles=[ln1, ln2], labels=["Train", "Valid"])

        plt.tight_layout()
        plt.savefig(self.save_path + "plots" + "/Loss_plot.png")

    def run(self, trainset, validset, testset):
        loss_record, valid_record = self.Train(trainset, validset)
