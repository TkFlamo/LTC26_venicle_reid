from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    dist = 1.0 - emb @ emb.T
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    pos = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return emb.sum() * 0
    return F.relu(pos[valid] - neg[valid] + margin).mean()


def supervised_contrastive(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    emb = F.normalize(emb, dim=-1)
    logits = emb @ emb.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    exp = torch.exp(logits).masked_fill(eye, 0)
    denom = exp.sum(dim=1).clamp_min(1e-8)
    log_prob = logits - torch.log(denom[:, None])
    npos = pos.sum(dim=1)
    valid = npos > 0
    if not valid.any():
        return emb.sum() * 0
    return -(log_prob.masked_fill(~pos, 0).sum(dim=1)[valid] / npos[valid]).mean()


def part_consistency(parts: torch.Tensor, visibility: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sim = torch.einsum("bpd,cpd->bcp", parts, parts)
    joint = visibility[:, None, :] & visibility[None, :, :]
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    pair = same & ~eye
    mask = joint & pair[:, :, None]
    if not mask.any():
        return parts.sum() * 0
    return (1.0 - sim[mask]).mean()


def part_supervised_contrastive(
    parts: torch.Tensor,
    visibility: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.08,
) -> torch.Tensor:
    losses = []
    for p in range(parts.shape[1]):
        keep = visibility[:, p].bool()
        if int(keep.sum()) < 3:
            continue
        # Need at least one positive identity pair in this part slot.
        lab = labels[keep]
        if not (lab[:, None].eq(lab[None, :]) & ~torch.eye(len(lab), device=lab.device, dtype=torch.bool)).any():
            continue
        losses.append(supervised_contrastive(parts[keep, p], lab, temperature=temperature))
    return torch.stack(losses).mean() if losses else parts.sum() * 0


def part_batch_hard_triplet(
    parts: torch.Tensor,
    visibility: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.18,
) -> torch.Tensor:
    losses = []
    for p in range(parts.shape[1]):
        keep = visibility[:, p].bool()
        if int(keep.sum()) < 3:
            continue
        lab = labels[keep]
        same = lab[:, None].eq(lab[None, :])
        if not (same & ~torch.eye(len(lab), device=lab.device, dtype=torch.bool)).any() or len(torch.unique(lab)) < 2:
            continue
        losses.append(batch_hard_triplet(parts[keep, p], lab, margin=margin))
    return torch.stack(losses).mean() if losses else parts.sum() * 0


def target_part_visibility(target: torch.Tensor, slot_ids: list[int], threshold: float = 0.25) -> torch.Tensor:
    # target: B,C,H,W (possibly soft after augmentation/downsampling)
    mass = target[:, slot_ids].flatten(2).amax(dim=-1)
    return mass >= float(threshold)


def weak_multilabel_part_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_supervision: torch.Tensor,
    available: torch.Tensor,
    quality: torch.Tensor,
    negative_weight: torch.Tensor,
    *,
    focal_gamma: float = 1.5,
    dice_weight: float = 0.45,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Quality-weighted multi-label semantic loss for polygon supervision.

    Positive polygons are trusted. The negative weight is configurable so the same loss can
    support either exhaustive labels (Carparts-Seg uses full negatives) or partial annotations.
    weight for its known ontology.  Overlapping vehicle/part masks remain legal.
    """
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="area")
    else:
        target = target.float()
    sup = class_supervision.float()[:, :, None, None]
    avail = available.float()[:, None, None, None]
    qual = quality.float().clamp(0.05, 1.0)[:, None, None, None]
    negw = negative_weight.float().clamp(0.0, 1.0)[:, None, None, None]

    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    prob = torch.sigmoid(logits.float())
    pt = target * prob + (1.0 - target) * (1.0 - prob)
    focal = (1.0 - pt).clamp_min(0).pow(float(focal_gamma))
    pixel_weight = target + (1.0 - target) * negw
    weight = sup * avail * qual * pixel_weight
    denom = weight.sum().clamp_min(1.0)
    bce_loss = (bce * focal * weight).sum() / denom

    # Positive-only soft Dice; missing classes do not create a Dice term.
    inter = (prob * target).sum(dim=(2, 3))
    denom_d = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom_d + 1.0)
    has_pos = target.sum(dim=(2, 3)) > 0.20
    valid = has_pos & class_supervision.bool() & available.bool()[:, None]
    if valid.any():
        q2 = quality[:, None].expand_as(dice).float()
        dice_loss = (dice[valid] * q2[valid]).sum() / q2[valid].sum().clamp_min(1e-6)
    else:
        dice_loss = logits.sum() * 0
    total = bce_loss + float(dice_weight) * dice_loss
    return total, {"part_bce": bce_loss.detach(), "part_dice": dice_loss.detach()}


def soft_batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    """Baseline V4 soft batch-hard triplet on Euclidean distance."""
    d = torch.cdist(emb.float(), emb.float())
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos = same & ~eye
    neg = ~same
    valid = pos.any(1) & neg.any(1)
    if not valid.any():
        return emb.sum() * 0
    hp = d.masked_fill(~pos, -1e9).max(1).values[valid]
    hn = d.masked_fill(~neg, 1e9).min(1).values[valid]
    return F.softplus(hp - hn + float(margin)).mean()


def circle_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.25, gamma: float = 64.0) -> torch.Tensor:
    z = F.normalize(emb, dim=-1)
    sim = z @ z.T
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pm = same & ~eye
    nm = ~same
    losses = []
    for i in range(len(labels)):
        sp = sim[i][pm[i]]; sn = sim[i][nm[i]]
        if sp.numel() == 0 or sn.numel() == 0:
            continue
        ap = torch.clamp_min(-sp.detach() + 1.0 + margin, 0.0)
        an = torch.clamp_min(sn.detach() + margin, 0.0)
        lp = -gamma * ap * (sp - (1.0 - margin))
        ln = gamma * an * (sn - margin)
        losses.append(F.softplus(torch.logsumexp(lp, 0) + torch.logsumexp(ln, 0)))
    return torch.stack(losses).mean() if losses else emb.sum() * 0


def cosine_distillation(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()


def part_probability_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Keep the Carparts semantic partition stable while late DINO blocks are fine-tuned.

    Use BCE-with-logits rather than ``binary_cross_entropy(sigmoid(...), target)``.
    The latter is explicitly unsafe under CUDA autocast (FP16/BF16) and aborts
    detail tuning before the first optimizer step.  Teacher probabilities are
    soft targets; gradients flow only through ``student_logits``.
    """
    t = max(float(temperature), 1e-3)
    student_scaled = student_logits.float() / t
    teacher_prob = torch.sigmoid(teacher_logits.detach().float() / t).clamp(1e-5, 1.0 - 1e-5)
    return F.binary_cross_entropy_with_logits(student_scaled, teacher_prob)


def encode_camera_ids(camera_ids, device: torch.device) -> torch.Tensor:
    """Encode arbitrary camera labels into a compact tensor for one batch."""
    lut = {}
    vals=[]
    for c in camera_ids:
        s=str(c)
        if s not in lut: lut[s]=len(lut)
        vals.append(lut[s])
    return torch.tensor(vals, device=device, dtype=torch.long)


def cross_camera_soft_batch_hard_triplet(
    emb: torch.Tensor,
    labels: torch.Tensor,
    cameras: torch.Tensor,
    margin: float = 0.15,
    cross_camera_weight: float = 0.75,
    any_camera_weight: float = 0.25,
) -> torch.Tensor:
    """Batch-hard metric loss aligned with the organizer junk protocol.

    The primary positive is the hardest image of the same identity from a *different* camera.
    Same-camera positives are only a regularizer.  Negatives include every different identity,
    including objects from the query camera, exactly as in the official evaluator.
    """
    d=torch.cdist(emb.float(),emb.float())
    same=labels[:,None].eq(labels[None,:])
    same_cam=cameras[:,None].eq(cameras[None,:])
    eye=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    neg=~same
    any_pos=same & ~eye
    cross_pos=same & ~same_cam
    hn=d.masked_fill(~neg,1e9).min(1).values
    terms=[]; weights=[]
    valid_cross=cross_pos.any(1)&neg.any(1)
    if valid_cross.any() and cross_camera_weight>0:
        hp=d.masked_fill(~cross_pos,-1e9).max(1).values
        terms.append(F.softplus(hp[valid_cross]-hn[valid_cross]+float(margin)).mean());weights.append(float(cross_camera_weight))
    valid_any=any_pos.any(1)&neg.any(1)
    if valid_any.any() and any_camera_weight>0:
        hp=d.masked_fill(~any_pos,-1e9).max(1).values
        terms.append(F.softplus(hp[valid_any]-hn[valid_any]+float(margin)).mean());weights.append(float(any_camera_weight))
    if not terms:return emb.sum()*0
    w=sum(weights); return sum(t*ww for t,ww in zip(terms,weights))/max(w,1e-12)


def cross_camera_supervised_contrastive(
    emb: torch.Tensor,
    labels: torch.Tensor,
    cameras: torch.Tensor,
    temperature: float = 0.07,
    cross_camera_positive_weight: float = 2.0,
    same_camera_positive_weight: float = 0.5,
) -> torch.Tensor:
    """SupCon where cross-camera positives contribute more than easy same-camera positives."""
    z=F.normalize(emb,dim=-1); logits=z@z.T/float(temperature)
    logits=logits-logits.max(1,keepdim=True).values.detach()
    eye=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    same=labels[:,None].eq(labels[None,:])&~eye
    diffcam=~cameras[:,None].eq(cameras[None,:])
    pos_w=torch.where(diffcam,torch.full_like(logits,float(cross_camera_positive_weight)),torch.full_like(logits,float(same_camera_positive_weight)))
    pos_w=pos_w*same.float()
    exp=torch.exp(logits).masked_fill(eye,0); denom=exp.sum(1).clamp_min(1e-8)
    logp=logits-torch.log(denom[:,None]); wsum=pos_w.sum(1); valid=wsum>0
    if not valid.any():return emb.sum()*0
    return -((logp*pos_w).sum(1)[valid]/wsum[valid]).mean()


class CrossBatchMemory:
    """FIFO memory bank used only as extra negatives; no stale-positive supervision."""
    def __init__(self, size: int = 8192):
        self.size=max(0,int(size)); self.emb=None; self.labels=None; self.cameras=None

    def __len__(self): return 0 if self.emb is None else int(self.emb.shape[0])

    @torch.no_grad()
    def update(self, emb: torch.Tensor, labels: torch.Tensor, cameras: torch.Tensor) -> None:
        if self.size<=0:return
        e=F.normalize(emb.detach().float(),dim=-1).cpu(); y=labels.detach().long().cpu(); c=cameras.detach().long().cpu()
        if self.emb is None:self.emb,self.labels,self.cameras=e,y,c
        else:
            self.emb=torch.cat([self.emb,e],0)[-self.size:]
            self.labels=torch.cat([self.labels,y],0)[-self.size:]
            self.cameras=torch.cat([self.cameras,c],0)[-self.size:]

    def tensors(self, device):
        if self.emb is None:return None
        return self.emb.to(device),self.labels.to(device),self.cameras.to(device)


def cross_batch_memory_triplet(
    emb: torch.Tensor,
    labels: torch.Tensor,
    cameras: torch.Tensor,
    memory: CrossBatchMemory | None,
    margin: float = 0.15,
) -> torch.Tensor:
    """Hardest current cross-camera positive versus hardest negative in a large FIFO memory."""
    if memory is None or len(memory)==0:return emb.sum()*0
    mt=memory.tensors(emb.device)
    if mt is None:return emb.sum()*0
    me,ml,_mc=mt
    dpos=torch.cdist(emb.float(),emb.float())
    same=labels[:,None].eq(labels[None,:]); diffcam=~cameras[:,None].eq(cameras[None,:])
    cross=same&diffcam
    valid_pos=cross.any(1)
    if not valid_pos.any():return emb.sum()*0
    hp=dpos.masked_fill(~cross,-1e9).max(1).values
    # Cosine distance is cheaper and stable for normalized memory embeddings.
    z=F.normalize(emb.float(),dim=-1); dm=1.0-z@me.T
    neg=~labels[:,None].eq(ml[None,:]); valid_neg=neg.any(1)
    hn=dm.masked_fill(~neg,1e9).min(1).values
    valid=valid_pos&valid_neg
    if not valid.any():return emb.sum()*0
    return F.softplus(hp[valid]-hn[valid]+float(margin)).mean()


def part_presence_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk: int = 4) -> torch.Tensor:
    """Augmentation-tolerant semantic self-training on target images.

    Spatial maps from strong/weak views are not aligned after flips/affine transforms, so compare
    robust top-k class presence rather than pixels. This adapts Carparts semantics without SAM3.
    """
    def presence(x):
        p=torch.sigmoid(x.float()).flatten(2); k=min(max(1,int(topk)),p.shape[-1]); return p.topk(k,dim=-1).values.mean(-1)
    return F.mse_loss(presence(student_logits),presence(teacher_logits.detach()))
