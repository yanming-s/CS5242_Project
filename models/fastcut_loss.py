import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

        target_real_label, target_fake_label = 1,0
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()


    def __call__(self, prediction,target_is_real):
        # prediction => (N,-1)
        bs = prediction.size(0)
        if self.gan_mode == "discriminator":
            target_tensor = self.real_label if target_is_real else self.fake_label
            target_tensor = target_tensor.expand_as(prediction) #Same shape as prediction
            loss = self.loss(prediction, target_tensor)
        else: # Non saturating (min Generator)
            if target_is_real:
                # Generator wants D(fake)=1
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)  
            else: 
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)

        return loss(prediction, target_tensor)
        

class PatchNCELoss(nn.Module):
    def __init__(self, options):
        super(PatchNCELoss, self).__init__()
        self.options = options
        self.tau =  0.07
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, feat_k, feat_q):
        '''
        feat_k : (N, C)
        feat_q : (N, C)
        '''

        num_patches, dim = feat_q.shape[0:2]

        # Dot product of same patch
        l_pos = feat_q.view(num_patches, 1, dim) @ feat_k.view(num_patches, dim, 1) # (num_patches, 1, 1)
        l_pos = l_pos.view(num_patches, 1) # (num_patches, 1)

        # For single-image translation task, negatives patches are all from same minibatch
        # For image-image translation task, negatives patches are from 1 sample in the minibatch
        B = 1 if self.options.iit_task else self.options.BS
        
        feat_q = feat_q.view(B, -1, dim)
        feat_k = feat_k.view(B, -1, dim)

        npatches = feat_q.size(1) # H*W
        
        l_neg_batch = feat_q @ feat_k.transpose(2, 1) #(B, H*W,C) @ (B, C, H*W) = (B, H*W, H*W) 

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :] # (-1, H*W, H*W)
        l_neg_batch.masked_fill_(diagonal, -10.0) #(B, H*W, H*W)  setting diagonals to ~0 

        l_neg = l_neg_batch.view(-1, npatches) #(B*H*W, H*W) in sit & (H*W) in iit
        out = torch.cat((l_pos, l_neg), dim=1) / self.tau #(B*H*W, H*W+1) -> Insert positives patches at 1st column

        loss = self.cross_entropy(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)) # #(B*H*W)
        return loss

        



