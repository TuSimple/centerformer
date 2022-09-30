import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class SegLoss(nn.Module):
  '''segmentation loss for an output tensor
    Arguments:
      mask (batch x dim x h x w)
      offset (batch x dim x h x w)
      gt_mask (batch x dim x h x w)
      gt_offset (batch x dim x h x w)
  '''
  def __init__(self, offset_weight =0.1):
    super(SegLoss, self).__init__()
    self.offset_weight = offset_weight
  
  def forward(self, mask, offset, gt_mask, gt_offset):
    loss = F.binary_cross_entropy(torch.sigmoid(mask), gt_mask)
    offset_loss = F.l1_loss(offset*gt_mask, gt_offset*gt_mask, reduction='none')
    offset_loss = offset_loss.sum() / (gt_mask.sum() + 1e-4)
    loss += self.offset_weight * offset_loss
    return loss

class SegLossV2(nn.Module):
  '''segmentation loss for an output tensor
    Arguments:
      mask (batch x dim x h x w)
      offset (batch x dim x h x w)
      grid_offset (batch x dim x h x w)
      gt_mask (batch x dim x h x w)
      gt_offset (batch x dim x h x w)
      gt_grid_offset (batch x dim x h x w)
  '''
  def __init__(self):
    super(SegLossV2, self).__init__()
  
  def forward(self, mask, offset, grid_offset, gt_mask, gt_offset, gt_grid_offset):
    loss = F.cross_entropy(mask, gt_mask.squeeze(1))
    offset_mask = (gt_mask>0).to(gt_offset)
    offset_loss = F.l1_loss(offset, gt_offset, reduction='none')*offset_mask
    offset_loss = offset_loss.sum() / (offset_mask.sum() + 1e-4)
    loss += offset_loss
    grid_offset_mask = (gt_mask==1).to(gt_offset)
    grid_offset_loss = F.l1_loss(F.sigmoid(grid_offset), gt_grid_offset, reduction='none')*grid_offset_mask
    grid_offset_loss = grid_offset_loss.sum() / (grid_offset_mask.sum() + 1e-4)
    loss += grid_offset_loss
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self, window_size=1, focal_factor=2):
    super(FastFocalLoss, self).__init__()
    self.window_size = window_size**2
    self.focal_factor = focal_factor

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, self.focal_factor) * gt
    neg_loss = neg_loss.sum()

    if self.window_size>1:
      ct_ind = ind[:,(self.window_size//2)::self.window_size]
      ct_mask = mask[:,(self.window_size//2)::self.window_size]
      ct_cat = cat[:,(self.window_size//2)::self.window_size]
    else:
      ct_ind = ind
      ct_mask = mask
      ct_cat = cat

    pos_pred_pix = _transpose_and_gather_feat(out, ct_ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, ct_cat.unsqueeze(2)) # B x M
    num_pos = ct_mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, self.focal_factor) * \
               ct_mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos
