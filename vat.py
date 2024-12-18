import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, framework="dgl", criterion=None, xi=1e-3, eps=2.5, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.framework = framework
        self.criterion = criterion

    def forward(self, model, x):
        if self.framework == "dgl":
            bg = x[0]
            nodefea = x[1]
            edgefea = x[2]

            with torch.no_grad():
                nodefea, edgefea = model.forwardProjector(nodefea, edgefea)
                pred, _ = model.forwardgnn(bg, nodefea, edgefea)

            # prepare random unit tensor
            dn = torch.rand(nodefea.shape).sub(0.5).to(nodefea.device)
            de = torch.rand(edgefea.shape).sub(0.5).to(edgefea.device)
            dn = _l2_normalize(dn)
            de = _l2_normalize(de)

            with _disable_tracking_bn_stats(model):
                # calc adversarial direction
                for _ in range(self.ip):
                    dn.requires_grad_()
                    de.requires_grad_()
                    pred_hat, _ = model.forwardgnn(
                        bg, nodefea + self.xi * dn, edgefea + self.xi * de
                    )
                    adv_distance = self.criterion(pred_hat, pred)
                    adv_distance = adv_distance.mean()
                    adv_distance.backward()
                    dn = _l2_normalize(dn.grad)
                    de = _l2_normalize(de.grad)
                    model.zero_grad()

                # calc LDS
                rn_adv = dn * self.eps
                re_adv = de * self.eps
                pred_hat, _ = model.forwardgnn(bg, nodefea + rn_adv, edgefea + re_adv)
                lds = self.criterion(pred_hat, pred)
        else:
            if self.framework == "geometric":
                z = x[0]
                pos = x[1]
                batch = x[2]
                # model(z, pos, batch)#[z, pos, batch]
                with torch.no_grad():
                    pred, _ = model(z, pos, batch)

                # prepare random unit tensor
                # dn = torch.rand(nodefea.shape).sub(0.5).to(nodefea.device)
                dn = torch.rand(pos.shape).sub(0.5).to(pos.device)
                # dn = _l2_normalize(dn)
                # de = _l2_normalize(de)

                with _disable_tracking_bn_stats(model):
                    # calc adversarial direction
                    for _ in range(self.ip):
                        dn.requires_grad_()
                        # de.requires_grad_()
                        pred_hat, _ = model(z, pos + self.xi * dn, batch)
                        adv_distance = self.criterion(pred_hat, pred)
                        adv_distance = adv_distance.mean()
                        # logp_hat = F.log_softmax(pred_hat, dim=1)
                        # adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                        adv_distance.backward()
                        dn = _l2_normalize(dn.grad)
                        # de = _l2_normalize(de.grad)
                        model.zero_grad()

                    # calc LDS
                    rn_adv = dn * self.eps
                    pred_hat, _ = model(z, pos + rn_adv, batch)
                    lds = self.criterion(pred_hat, pred)

            elif self.framework == "mat":
                data = x  # `data` is the full input to the DEEP_GATGNN model

                # Store original node features and edge attributes
                original_x = data.x.clone().detach()
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    original_edge_attr = data.edge_attr.clone().detach()
                else:
                    original_edge_attr = None

                with torch.no_grad():
                    pred, _ = model(data)

                # Prepare random unit tensor for node features
                dn = torch.rand_like(data.x).sub(0.5)
                dn = _l2_normalize(dn)

                # Prepare random unit tensor for edge features if available
                if original_edge_attr is not None:
                    de = torch.rand_like(data.edge_attr).sub(0.5)
                    de = _l2_normalize(de)
                else:
                    de = None

                with _disable_tracking_bn_stats(model):
                    # Calc adversarial direction
                    for _ in range(self.ip):
                        dn.requires_grad_()
                        if de is not None:
                            de.requires_grad_()

                        # Perturb node features and edge attributes
                        data.x = original_x + self.xi * dn
                        if de is not None:
                            data.edge_attr = original_edge_attr + self.xi * de

                        pred_hat, _ = model(data)
                        adv_distance = self.criterion(pred_hat, pred)
                        adv_distance = adv_distance.mean()
                        adv_distance.backward()

                        dn = _l2_normalize(dn.grad)
                        if de is not None:
                            de = _l2_normalize(de.grad)

                        model.zero_grad()

                    # Calc LDS
                    rn_adv = dn * self.eps
                    if de is not None:
                        re_adv = de * self.eps

                    data.x = original_x + rn_adv
                    if de is not None:
                        data.edge_attr = original_edge_attr + re_adv

                    pred_hat, _ = model(data)
                    lds = self.criterion(pred_hat, pred)

                    # Reset data to original state
                    data.x = original_x
                    if de is not None:
                        data.edge_attr = original_edge_attr
        return lds
