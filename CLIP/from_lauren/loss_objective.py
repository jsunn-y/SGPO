import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Temperature hyperparameter
LOGIT_SCALE = 0.07 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class Lcon(nn.Module):
    def __init__(self, logit_scale=LOGIT_SCALE):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/logit_scale), dtype=torch.float32))

    def forward(self, image_features, text_features):
        b_img = image_features.size(0)
        b_txt = text_features.size(0)
        b     = min(b_img, b_txt)

        if b_img != b_txt:
            image_features = image_features[:b]
            text_features  = text_features [:b]

        image_features = F.normalize(image_features, dim=1)
        text_features  = F.normalize(text_features,  dim=1)

        s = self.logit_scale.exp().clamp(1e-3, 1e3)
        logits_per_image = s * (image_features @ text_features.t())  # [b, b]
        logits_per_text  = logits_per_image.t()                     # [b, b]

        device   = image_features.device
        labels_i = torch.arange(b, device=device)
        labels_t = torch.arange(b, device=device)

        loss_i = F.cross_entropy(logits_per_image, labels_i)
        loss_t = F.cross_entropy(logits_per_text,  labels_t)
        return 0.5 * (loss_i + loss_t)

class LconNEG(nn.Module):
    def __init__(self, logit_scale=LOGIT_SCALE):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/logit_scale), dtype=torch.float32))

    def forward(self, image_features, pos_text_features, neg_text_features):
        b_img = image_features.size(0)
        b_pos = pos_text_features.size(0)
        b_neg = neg_text_features.size(0)
        b = min(b_img, b_pos, b_neg)

        # 2) truncate all three to length b
        image_features = image_features[:b]
        pos_text_features = pos_text_features[:b]
        neg_text_features = neg_text_features[:b]

        # 3) normalize
        image_features    = F.normalize(image_features,    dim=1)
        pos_text_features = F.normalize(pos_text_features, dim=1)
        neg_text_features = F.normalize(neg_text_features, dim=1)

        # 4) compute logits
        s = self.logit_scale.exp().clamp(1e-3, 1e3)
        # image→all‑text (2b candidates)
        all_text = torch.cat([pos_text_features, neg_text_features], dim=0)  # [2b, d]
        logits_per_image = s * (image_features @ all_text.t())               # [b, 2b]
        # positive‑text→image (b candidates)
        logits_per_text  = s * (pos_text_features @ image_features.t())      # [b,  b]

        # 5) build labels from each “row count”
        device   = image_features.device
        labels_i = torch.arange(logits_per_image.size(0), device=device)
        labels_t = torch.arange(logits_per_text .size(0), device=device)

        # 6) average the two cross‑entropies
        loss_i = F.cross_entropy(logits_per_image, labels_i)
        loss_t = F.cross_entropy(logits_per_text,  labels_t)
        return 0.5 * (loss_i + loss_t)

#focus on this part accoridng to lauren
class Lspread(nn.Module):
    def __init__(self, logit_scale=LOGIT_SCALE, alpha=0.5):
        super(Lspread, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))
        self.logit_scale.data.clamp_(np.log(1/100), np.log(100))  # keep in [0.01,100]
        self.alpha = alpha

    def forward(self, img_embs, pos_txt_embs, neg_txt_embs):
        img_embs = img_embs / img_embs.norm(dim=1, keepdim=True)
        pos_txt_embs = pos_txt_embs / pos_txt_embs.norm(dim=1, keepdim=True)
        neg_txt_embs = neg_txt_embs / neg_txt_embs.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp().clamp(1e-3, 1e3)

        logits_per_img_pos = logit_scale * img_embs @ pos_txt_embs.t()  # [n, n]
        logits_per_img_neg = logit_scale * img_embs @ neg_txt_embs.t()  # [n, n]
        # VL: CONCON: numerator: similar context logits for image (positive + negative txt embs with the same index) 
        sim_context_logits_img = torch.cat([logits_per_img_pos.unsqueeze(1), logits_per_img_neg.unsqueeze(1)], dim=1)
        # VL: CONCON: denom: all text logits
        all_txt_embs = torch.cat([pos_txt_embs, neg_txt_embs], dim=0)  # [2n, d]
        logits_per_img_all = logit_scale * img_embs @ all_txt_embs.t()  # [n, 2n]
        concon_vl_loss = -torch.mean(torch.log(
            torch.sum(torch.exp(sim_context_logits_img), dim=1) /
            torch.sum(torch.exp(logits_per_img_all), dim=1)
        ))
        # VL: contextNCE: to separate "within similar context: txt embs 
        contextnce_vl_loss = -torch.log(torch.exp(logits_per_img_pos) / torch.sum(torch.exp(sim_context_logits_img), dim=1, keepdim=True)).mean()
        L_spread_vl = (1 - self.alpha) * concon_vl_loss + self.alpha * contextnce_vl_loss

        # LV: No correct image for negative text. So, just contrastive approach since we consider only positive text case. 
        logits_per_text = logit_scale * pos_txt_embs @ img_embs.t()  # [n, n]
        cont_lv_loss = nn.CrossEntropyLoss()(logits_per_text, torch.arange(logits_per_text.size(0)).to(device))

        # Total Loss
        L_spread = 0.5(L_spread_vl + cont_lv_loss)
        return L_spread


class LspreadMOD(nn.Module):
    def __init__(self, logit_scale=LOGIT_SCALE, alpha=0.7, K=8):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))
        self.logit_scale.data.clamp_(np.log(1/100), np.log(100))
        self.alpha = alpha  
        self.K = K  # number of extra neighbors to incorporate in similar context size 

    def forward(self, img_embs, pos_txt_embs, neg_txt_embs):
        img_embs     = img_embs     / img_embs.norm(dim=1, keepdim=True)
        pos_txt_embs = pos_txt_embs / pos_txt_embs.norm(dim=1, keepdim=True)
        neg_txt_embs = neg_txt_embs / neg_txt_embs.norm(dim=1, keepdim=True)

        t = self.logit_scale.exp().clamp(1e-3, 1e3)

        all_txt = torch.cat([pos_txt_embs, neg_txt_embs], dim=0)   # [2n, d]
        logits_all = t * (img_embs @ all_txt.t())                  # [n, 2n]
        n = img_embs.size(0)
        device = img_embs.device

        # Indices of matching pos/neg in txt logits bank [2n]
        pos_idx = torch.arange(n, device=device)    # 0..n-1
        neg_idx = pos_idx + n   # n..2n-1

        # Find topK other positives (mask self‐match)
        with torch.no_grad():
            cospos = pos_txt_embs @ pos_txt_embs.t()    # [n,n]
            eye = torch.eye(n, dtype=torch.bool, device=device)
            cospos.masked_fill_(eye, -9e9)
            _, topk = cospos.topk(self.K, dim=1)    # [n, K]

        # Each row’s similar context indices
        # col0 = pos_idx, col1 = neg_idx, col2.. = topK nearest pos
        all_pos_idx = torch.cat([pos_idx.unsqueeze(1), neg_idx.unsqueeze(1), topk], dim=1)  # [n, 2+K]

        # VL: CONCON
        exp_all = logits_all.exp()  # [n, 2n]
        num = exp_all.gather(1, all_pos_idx).sum(dim=1) # [n]
        denom = exp_all.sum(dim=1)    # [n]
        concon_vl_loss = -torch.log(num/denom).mean()

        # VL: contextNCE: single‐out the “true” positive (first column)
        logits_context = logits_all.gather(1, all_pos_idx)     # [n, 2+K]
        # log‐softmax over the similar context set, then pick col0
        logprob = F.log_softmax(logits_context, dim=1)[:, 0]   # [n]
        contextnce_vl_loss = -logprob.mean()
        L_spread_vl = (1 - self.alpha) * concon_vl_loss + self.alpha * contextnce_vl_loss

        # LV: standard contrastive approach over pos_txt to img)
        logits_per_text = t * (pos_txt_embs @ img_embs.t())     # [n, n]
        cont_lv_loss = nn.CrossEntropyLoss()(logits_per_text, torch.arange(n, device=device))

        L_spread = L_spread_vl + cont_lv_loss
        return L_spread
