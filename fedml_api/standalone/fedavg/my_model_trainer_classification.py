import torch
from torch import nn
from tqdm import tqdm
from vat import VATLoss
import torch.nn.functional as F
from utils import mybceloss

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args, argsinput, _data):
        super().__init__(model)
        self.args = args
        self.wandbconfig = argsinput
        self.data = _data  # Store data for use in the model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, roundidx, clientidx):

        fedmid = self.wandbconfig.fedmid
        weightReg = self.wandbconfig.weightReg
        tmpFed = self.wandbconfig.tmpFed
        lambdavat = self.wandbconfig.lambdavat
        warmupRound = 1

        model = self.model
        model.to(device)
        model.train()
        # init a new model
        if args.dataset == "qm9":
            from network.myschnet import SchNet

            globalModel = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=10.0,
            )
            lossCriterion = nn.L1Loss(reduction="none")

        elif args.dataset in [
            "mat",
            "band",
            "formation",
            "2d",
            "alloy",
            "pt",
            "dielecric",
            "elasticity",
            "perovskites",
        ]:
            from matdeeplearn.models.deep_gatgnn import DEEP_GATGNN
            from matdeeplearn.models.schnet import SchNet as SchNet_Mat
            from matdeeplearn.models.mpnn import MPNN

            data2 = self.data

            if args.model == "deepergatgnn":
                globalModel = DEEP_GATGNN(
                    data=data2,
                    dim1=64,
                    dim2=150,
                    pre_fc_count=1,
                    gc_count=20,
                    post_fc_count=0,
                    pool="global_add_pool",
                    pool_order="early",
                    batch_norm="True",
                    batch_track_stats="True",
                    act="softplus",
                    dropout_rate=0.0,
                )
            elif args.model == "schnet":
                globalModel = SchNet_Mat(
                    data=data2,
                    dim1=64,
                    dim2=100,
                    dim3=150,
                    pre_fc_count=1,
                    gc_count=4,
                    post_fc_count=3,
                    pool="global_max_pool",
                    pool_order="early",
                    batch_norm="True",
                    batch_track_stats="True",
                    act="relu",
                    dropout_rate=0.0,
                )
            elif args.model == "mpnn":
                globalModel = MPNN(
                    data=data2,
                    dim1=64,
                    dim2=100,
                    dim3=100,
                    pre_fc_count=1,
                    gc_count=4,
                    post_fc_count=3,
                    pool="global_mean_pool",
                    pool_order="early",
                    batch_norm="True",
                    batch_track_stats="True",
                    act="relu",
                    dropout_rate=0.0,
                )

            lossCriterion = nn.L1Loss(reduction="none")

        elif args.dataset in ["esol", "lipo", "freesolv"]:
            from network.myMPNNPredictor import MPNNPredictor

            globalModel = MPNNPredictor(
                node_in_feats=train_data.dataset.dataset.dataset.graphs[0]
                .ndata["h"]
                .shape[1],
                edge_in_feats=train_data.dataset.dataset.dataset.graphs[0]
                .edata["e"]
                .shape[1],
                node_out_feats=64,
                edge_hidden_feats=16,
                num_step_message_passing=3,
                num_step_set2set=3,
                num_layer_set2set=3,
                n_tasks=1,
            )
            lossCriterion = nn.MSELoss(reduction="none")
        elif args.dataset in [
            "MUV",
            "BACE",
            "BBBP",
            "ClinTox",
            "SIDER",
            "ToxCast",
            "HIV",
            "PCBA",
            "Tox21",
        ]:
            from network.myMPNNPredictor import MPNNPredictor

            globalModel = MPNNPredictor(
                node_in_feats=train_data.dataset.dataset.dataset.graphs[0]
                .ndata["h"]
                .shape[1],
                edge_in_feats=train_data.dataset.dataset.dataset.graphs[0]
                .edata["e"]
                .shape[1],
                node_out_feats=64,
                edge_hidden_feats=16,
                num_step_message_passing=3,
                num_step_set2set=3,
                num_layer_set2set=3,
                n_tasks=train_data.dataset.dataset.dataset.n_tasks,
            )
            lossCriterion = mybceloss()
        else:
            raise ValueError("not found dataset")
        globalModel = globalModel.to(device)
        for param_q, param_k in zip(model.parameters(), globalModel.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = True  # not update by gradient

        if args.dataset in [
            "mat",
            "band",
            "formation",
            "2d",
            "alloy",
            "pt",
            "dielecric",
            "elasticity",
            "perovskites",
        ]:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0001, weight_decay=1e-5
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.0001, weight_decay=1e-5
            )
        localsteps = args.localStepsPerRound
        tmpstep = 0

        predGAll, predGAll_emb, predGAll_vat = self.globalEpoch(
            train_data, globalModel, args, device, lossCriterion
        )
        weight_denomaitor = None
        tbar = tqdm(train_data, mininterval=2, disable=True)
        while tmpstep < localsteps:
            tbarLocalAll = tqdm(range(args.epochs))
            for epoch in enumerate(tbarLocalAll):
                batch_loss = []
                for batch_idx, data in enumerate(tbar):
                    if tmpstep >= localsteps:
                        if fedmid == "moon":
                            embedding = self.globalEpoch(
                                train_data, self.model, args, device, lossCriterion
                            )
                            if hasattr(self, "embedding"):
                                self.embedding[clientidx] = embedding
                            else:
                                self.embedding = {}
                                self.embedding[clientidx] = embedding
                        del globalModel
                        return

                    tmpstep += 1
                    optimizer.zero_grad()
                    if args.dataset == "qm9":
                        z, pos, batch, labels = (
                            data.z.to(device),
                            data.pos.to(device),
                            data.batch.to(device),
                            data.y.to(device),
                        )
                        if "vat" in fedmid:
                            vatloss = VATLoss(
                                framework="geometric",
                                criterion=lossCriterion,
                                xi=args.xi,
                            )  # xi, and eps
                            xCombined = [z, pos, batch]
                            localVATLoss = vatloss(model, xCombined)

                        pred, latentEmb = model(z, pos, batch)
                        index = data.idx
                        predG = predGAll[index]
                        latentEmbG = predGAll_emb[index]
                        globalVATLoss = predGAll_vat[index]

                    elif args.dataset in [
                        "mat",
                        "band",
                        "formation",
                        "2d",
                        "alloy",
                        "pt",
                        "dielecric",
                        "elasticity",
                        "perovskites",
                    ]:
                        data = data.to(device)
                        labels = data.y.to(device)
                        index = data.idx.to(device)

                        if "vat" in fedmid:
                            vatloss = VATLoss(
                                framework="mat",
                                criterion=lossCriterion,
                                xi=args.xi,
                            )
                            localVATLoss = vatloss(model, data)

                        pred, latentEmb = model(data)
                        predG = predGAll[index]
                        latentEmbG = predGAll_emb[index]
                        globalVATLoss = predGAll_vat[index]

                    elif args.dataset in [
                        "esol",
                        "lipo",
                        "freesolv",
                        "MUV",
                        "BACE",
                        "BBBP",
                        "ClinTox",
                        "SIDER",
                        "ToxCast",
                        "HIV",
                        "PCBA",
                        "Tox21",
                    ]:
                        smiles, bg, labels, index = data
                        if len(smiles) == 1:
                            continue
                        labels, index = labels.to(device), index.to(device)
                        bg = bg.to(device)
                        node_feats = bg.ndata.pop("h").to(device)
                        edge_feats = bg.edata.pop("e").to(device)

                        if "vat" in fedmid:
                            vatloss = VATLoss(
                                framework="dgl", criterion=lossCriterion, xi=args.xi
                            )  # xi, and eps
                            xCombined = [bg, node_feats, edge_feats]
                            localVATLoss = vatloss(model, xCombined)

                        pred, latentEmb = model(bg, node_feats, edge_feats)
                        predG = predGAll[index]
                        latentEmbG = predGAll_emb[index]
                        globalVATLoss = predGAll_vat[index]

                    if roundidx < warmupRound:
                        loss = lossCriterion(pred, labels)
                        loss = loss.mean()
                    else:
                        if fedmid == "avg":
                            loss = lossCriterion(pred, labels)
                            loss = loss.mean()

                        elif fedmid == "oursvatFLITPLUS":
                            lossGlobalLabel = lossCriterion(predG, labels)
                            lossLocalLabel = lossCriterion(pred, labels)
                            lossLocalVAT = localVATLoss
                            lossGlobalVAT = globalVATLoss

                            weightloss_loss = lossLocalLabel + torch.relu(
                                lossLocalLabel - lossGlobalLabel.detach()
                            )
                            weightloss_vat = localVATLoss + torch.relu(
                                lossLocalVAT - lossGlobalVAT.detach()
                            )
                            weightloss = weightloss_loss + lambdavat * weightloss_vat
                            factor_ema = 0.8
                            if weight_denomaitor == None:
                                weight_denomaitor = weightloss.mean(
                                    dim=0, keepdim=True
                                ).detach()
                            else:
                                weight_denomaitor = (
                                    factor_ema * weight_denomaitor
                                    + (1 - factor_ema)
                                    * weightloss.mean(dim=0, keepdim=True).detach()
                                )
                            loss = (
                                1
                                - torch.exp(-weightloss / (weight_denomaitor + 1e-7))
                                + 1e-7
                            ) ** tmpFed * (lossLocalLabel + weightReg * lossLocalVAT)

                            loss = loss.mean()

                        elif fedmid == "oursFLIT":
                            lossGlobalLabel = lossCriterion(predG, labels)
                            lossLocalLabel = lossCriterion(pred, labels)

                            weightloss = lossLocalLabel + torch.relu(
                                lossLocalLabel - lossGlobalLabel.detach()
                            )
                            factor_ema = 0.8
                            if weight_denomaitor == None:
                                weight_denomaitor = weightloss.mean(
                                    dim=0, keepdim=True
                                ).detach()
                            else:
                                weight_denomaitor = (
                                    factor_ema * weight_denomaitor
                                    + (1 - factor_ema)
                                    * weightloss.mean(dim=0, keepdim=True).detach()
                                )

                            loss = (
                                1
                                - torch.exp(-weightloss / (weight_denomaitor + 1e-7))
                                + 1e-7
                            ) ** tmpFed * (lossLocalLabel)
                            loss = loss.mean()
                        else:
                            print(fedmid)
                            raise ValueError("not found fed method")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    batch_loss.append(loss.item())
                    del loss

    def globalEpoch(self, train_data, globalModel, args, device, criterion):
        # globalModel.eval()
        if args.dataset == "qm9":
            # print(type(train_data.dataset))
            # exit()
            numData = train_data.dataset.data.idx[-1]
            numTasks = train_data.dataset.data.y.shape[-1]
            embed_dim = 128

        elif args.dataset in [
            "mat",
            "band",
            "formation",
            "2d",
            "alloy",
            "pt",
            "dielecric",
            "elasticity",
            "perovskites",
        ]:
            numData = len(train_data.dataset.dataset.dataset)
            numTasks = train_data.dataset.dataset.dataset.y.shape[-1]
            # numTasks = train_data.dataset.dataset.dataset.y.size(-1)
            embed_dim = 64
        else:
            # print(type(train_data.dataset))
            # exit()
            numData = len(train_data.dataset.dataset.dataset)
            numTasks = train_data.dataset.dataset.dataset.labels.shape[-1]
            embed_dim = 128
        predGAll = torch.zeros(numData, numTasks).cuda()
        predGAll_loss = torch.zeros(numData, numTasks).cuda()
        predGAll_vat = torch.zeros(numData, numTasks).cuda()
        predGAll_emb = torch.zeros(numData, embed_dim).cuda()
        tbar1 = tqdm(train_data)

        for batch_idx, data in enumerate(tbar1):
            if args.dataset == "qm9":
                z, pos, batch, labels = (
                    data.z.to(device),
                    data.pos.to(device),
                    data.batch.to(device),
                    data.y.to(device),
                )
                x = [z, pos, batch]
                index = data.idx

                globalModel.zero_grad()
                if True:
                    vatloss = VATLoss(
                        framework="geometric", criterion=criterion, xi=args.xi
                    )  # xi, and eps
                    predGAll_vat[index.squeeze()] = vatloss(
                        globalModel.train(), x
                    ).detach()
                with torch.no_grad():
                    predG, latentEmbG = globalModel(z, pos, batch)
                    # print(f"shape of latentEmbG: {latentEmbG.shape}")
                    losstmp = criterion(predG, labels)

                    predGAll_loss[index.squeeze()] = losstmp.detach()
                    predGAll[index.squeeze()] = predG.detach()
                    # print(index.squeeze().shape, "xxxxxxxxxx")
                    predGAll_emb[index.squeeze()] = latentEmbG.clone().detach()

            elif args.dataset in [
                "mat",
                "band",
                "formation",
                "2d",
                "alloy",
                "pt",
                "dielecric",
                "elasticity",
                "perovskites",
            ]:
                data = data.to(device)
                index = data.idx

                globalModel.zero_grad()
                if True:
                    vatloss = VATLoss(
                        framework="mat", criterion=criterion, xi=args.xi
                    )  # xi, and eps
                    predGAll_vat[index.squeeze()] = (
                        vatloss(globalModel.train(), data).detach().view(-1, 1)
                    )
                with torch.no_grad():
                    predG, latentEmbG = globalModel(data)

                losstmp = criterion(predG, data.y)
                predGAll_loss[index] = losstmp.detach().view(-1, 1)
                predGAll[index] = predG.detach().view(-1, 1)
                predGAll_emb[index] = latentEmbG.clone().detach()
                # print("xxxxx")
                # exit()

            elif args.dataset in [
                "esol",
                "lipo",
                "freesolv",
                "MUV",
                "BACE",
                "BBBP",
                "ClinTox",
                "SIDER",
                "ToxCast",
                "HIV",
                "PCBA",
                "Tox21",
            ]:
                smiles, bg, labels, index = data
                labels, index = labels.to(device), index.to(device)
                bg = bg.to(device)
                node_feats = bg.ndata.pop("h").to(device)
                edge_feats = bg.edata.pop("e").to(device)
                globalModel.zero_grad()
                x = [bg, node_feats, edge_feats]
                if True:
                    vatloss = VATLoss(
                        framework="dgl", criterion=criterion, xi=args.xi
                    )  # xi, and eps
                    predGAll_vat[index.squeeze()] = vatloss(globalModel, x).detach()

                with torch.no_grad():
                    predG, latentEmbG = globalModel(bg, node_feats, edge_feats)
                    print(f"shape of latentEmbG: {latentEmbG.shape}")

                losstmp = criterion(predG, labels)
                predGAll_loss[index.squeeze()] = losstmp.detach()
                predGAll[index.squeeze()] = predG.detach()
                predGAll_emb[index.squeeze()] = latentEmbG.clone().detach()
        return predGAll, predGAll_emb, predGAll_vat
