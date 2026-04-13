"""
Created on Sep 15th 18:05:53 2020
Author: Qizhong Zhang
"""

import argparse
import csv
import datetime
import os

import numpy as np
import torch
from torch.utils.data import Subset

from data_prepare import MoveSimData, reform
from model import VAE
from result_evaluation import EVALUATION

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class parameters(object):

    def __init__(self, args) -> None:
        super().__init__()

        # Data-related
        self.data_type = args.data_type
        self.location_mode = args.location_mode
        self.trainsize = args.trainsize

        # Model-related
        self.rnn_type = args.rnn_type
        self.rnn_layers = args.rnn_layers
        self.rnn_bidirectional = args.rnn_bidirectional
        self.dual_rnn = args.dual_rnn
        self.tim_emb_type = args.tim_emb_type
        self.tim_emb_size = args.tim_emb_size
        self.loc_emb_size = args.loc_emb_size
        self.usr_emb_size = args.usr_emb_size
        self.d_model = args.d_model
        self.encoder_rnn_hidden_size = args.encoder_rnn_hidden_size
        self.z_hidden_size_mean = args.z_hidden_size_mean
        self.z_hidden_size_std = args.z_hidden_size_std
        self.latent_size = args.latent_size
        self.layernorm = args.layernorm
        self.dropout = args.dropout
        self.decoder_rnn_hidden_size = args.decoder_rnn_hidden_size
        self.loc_hidden_size1 = args.loc_hidden_size1
        self.loc_hidden_size2 = args.loc_hidden_size2
        self.cdfpoi = args.cdfpoi
        self.poi_weight = args.poi_weight
        self.poi_weight_dynamic = args.poi_weight_dynamic
        self.loc_initial = args.loc_initial
        self.tim_hidden_size1 = args.tim_hidden_size1
        self.tim_hidden_size2 = args.tim_hidden_size2
        self.time_initial = args.time_initial

        self.max_beta = args.max_beta
        self.cycle = int(args.cycle * 32 / args.batchsize)
        self.learning_rate = args.learning_rate
        self.L2 = args.L2
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.epoches = args.epoches
        self.batchsize = args.batchsize
        self.ntrajs = args.ntrajs
        self.first_sample = args.first_sample
        self.tim_only = args.tim_only
        self.loc_only = args.loc_only
        self.fourier = args.fourier
        self.poi_ban = args.poi_ban
        self.poi_emb_ban = args.poi_emb_ban
        self.pos_emb_ban = args.pos_emb_ban
        self.feedback_ban = args.feedback_ban

        self.save_path = "./RES/" + str(datetime.datetime.now().strftime("%Y-%m%d-%H%M") + "/0/")
        self.param_name = [x for x in self.__dict__]
        self.param_value = [self.__dict__[v] for v in self.__dict__]

        self.tune = args.tune
        self.exptimes = args.exptimes
        self.checkpoint = args.checkpoint
        self.generate = args.generate

        self.cuda = args.cuda
        self.device = torch.device(("cuda:" + args.cuda) if torch.cuda.is_available() else "cpu")

    def data_info(self, data):
        self.POI = data.POI
        self.GPS = data.GPS
        self.USERLIST = data.USERLIST
        self.loc_size = data.loc_size
        self.tim_size = data.tim_size
        self.usr_size = data.usr_size
        self.poi_size = data.poi_size
        self.infer_maxlast = data.infer_maxlast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data-related
    parser.add_argument("-d", "--data_type", type=str, required=True, choices=["San_Francisco", "Porto", "Beijing"])
    parser.add_argument("-l", "--location_mode", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("-t", "--trainsize", type=float, default=0.9)

    # Model-related
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_bidirectional", type=bool, default=False)
    parser.add_argument("--dual_rnn", type=bool, default=False)
    parser.add_argument("--tim_emb_type", type=str, default="Linear", choices=["Linear", "Categorical"])
    parser.add_argument("--tim_emb_size", type=int, default=256)
    parser.add_argument("--loc_emb_size", type=int, default=256)
    parser.add_argument("--usr_emb_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--encoder_rnn_hidden_size", type=int, default=512)
    parser.add_argument("--z_hidden_size_mean", type=int, default=256)
    parser.add_argument("--z_hidden_size_std", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--layernorm", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decoder_rnn_hidden_size", type=int, default=512)
    parser.add_argument("--loc_hidden_size1", type=int, default=128)
    parser.add_argument("--loc_hidden_size2", type=int, default=128)
    parser.add_argument("--cdfpoi", action="store_false")
    parser.add_argument("--poi_weight", type=float, default=0.1)
    parser.add_argument("--poi_weight_dynamic", type=bool, default=False)
    parser.add_argument("--loc_initial", type=float, default=1e9)
    parser.add_argument("--tim_hidden_size1", type=int, default=128)
    parser.add_argument("--tim_hidden_size2", type=int, default=128)
    parser.add_argument("--time_initial", type=float, default=100)

    parser.add_argument("--max_beta", type=float, default=0.5)
    parser.add_argument("--cycle", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--L2", type=float, default=1e-5)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("-e", "--epoches", type=int, default=50)
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("--tim_only", type=bool, default=False)
    parser.add_argument("--loc_only", type=bool, default=False)
    parser.add_argument("--ntrajs", type=int, default=7)
    parser.add_argument("--first_sample", type=str, default="New", choices=["New", "Original"])
    parser.add_argument("--fourier", type=bool, default=False)
    parser.add_argument("--poi_ban", type=bool, default=False)
    parser.add_argument("--poi_emb_ban", type=bool, default=False)
    parser.add_argument("--pos_emb_ban", type=bool, default=False)
    parser.add_argument("--feedback_ban", type=bool, default=False)

    parser.add_argument("-n", "--exptimes", type=int, default=1)
    parser.add_argument("--tune", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--generate", type=bool, default=False)
    parser.add_argument("--cuda", type=str, default="0", choices=["0", "1", "2", "3"])

    args = parser.parse_args()
    param = parameters(args)

    data = MoveSimData(param.data_type)
    param.data_info(data)

    trainid, validid, testid = data.split()
    trainset = Subset(data, trainid)
    validset = Subset(data, validid)
    testset = Subset(data, testid)

    reform(trainset, "train")
    reform(testset, "test")

    # Logging
    os.makedirs(param.save_path[:-2], exist_ok=True)
    with open(param.save_path[:-2] + "result.csv", "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(param.param_name)
        csv_writer.writerow(param.param_value)
        csv_writer.writerow(["Method", "travel_distance", "radius", "duration", "G_rank", "move", "stay"])

    for i in range(param.exptimes):
        print("Data Loaded")

        param.save_path = param.save_path[:-2] + str(i) + "/"
        os.makedirs(param.save_path + "plots")
        os.makedirs(param.save_path + "data")

        model = VAE(param)
        model = model.float().to(param.device)
        if param.checkpoint is not None:
            model.load(param.checkpoint)

        if param.generate:
            model.initial_prob = model.inference_initial(trainset)
            model.user_indicator, model.loc_weights = model.location_constraints(trainset.dataset.REFORM["train"])
            model._cache_indicator_tensor()

            all_lens = [len(traj["loc"]) for trajs in data.REFORM["test"].values() for traj in trajs.values()]
            model.infer_maxlast = max(all_lens)
            model.ntrajs = 1
            model.test_data_prepare_od(testset.dataset)
        else:
            model.run(trainset, validset, testset)

        np.save(param.save_path + "data/original.npy", data.REFORM["test"])
