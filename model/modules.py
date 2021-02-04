import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def get_text_pooling(fusion_type, *args, **kwargs):
    Fusion = {
        'weightedconv': TextWeightedConv,
        'sharedconv': SharedConv,
        'singleconv': SharedConv,
        'sharedfc': SharedFC,
        'sharedgru': SharedGRU,
    }[fusion_type.lower()]
    return Fusion(*args, **kwargs)

class TextWeightedConv(nn.Module):
    def __init__(self, word_dim, hidden_dim, **kwargs):
        super(TextWeightedConv, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv1d(word_dim, hidden_dim, 2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fc = torch.nn.Sequential(
            nn.Linear(word_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.att = nn.Linear(word_dim+hidden_dim, 1)
        self.out_dim = hidden_dim
        
    def forward(self, text, text_lengths):
        '''
        args
          text: B x L x wD
          text_lenghts: list of int (B)
        '''
        # Build mask
        B, L, _ = text.size()
        t_len = torch.tensor(text_lengths)
        mask = torch.arange(L).expand(B,L) < t_len.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=text.dtype, device=text.device) # B x L
        
        # Conv
        con_out = text.transpose(1,2) #  B x wD x L
        con_out = self.conv(con_out)[:,:,:-1] #  B x D x L
        con_out = con_out.transpose(1,2) # B x L x D
        con_out = torch.cat([con_out,text], 2) 
            
        con_feat = self.fc(con_out) 
        con_feat = con_feat * mask.unsqueeze(-1) # B x L x D
        
        con_att = self.att(con_out) # B x L x 1
        con_att = F.pad(con_att.squeeze(-1), (1,0))
        con_att = masked_softmax(con_att, F.pad(mask, (1,0), 'constant', 1)) # B x (L + 1)
        con_att = con_att[:,1:].unsqueeze(-1) # B x L x 1
        
        feature = (con_feat * con_att).sum(1) # B x D
        
        return feature

class SharedConv(nn.Module):
    def __init__(self, word_dim, hidden_dim, **kwargs):
        super(SharedConv, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv1d(word_dim, hidden_dim, 2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fc = torch.nn.Sequential(
            nn.Linear(word_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.out_dim = hidden_dim
        
    def forward(self, text, text_lengths):
        '''
        args
          text: B x L x wD
          text_lenghts: list of int (B)
        '''
        # Build mask
        B, L, _ = text.size()
        t_len = torch.tensor(text_lengths)
        mask = torch.arange(L).expand(B,L) < t_len.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=text.dtype, device=text.device) # B x L
        
        # Conv
        con_out = text.transpose(1,2) #  B x wD x L
        con_out = self.conv(con_out)[:,:,:-1] #  B x D x L
        con_out = con_out.transpose(1,2) # B x L x D
        con_out = torch.cat([con_out,text], 2) 
            
        con_feat = self.fc(con_out) 
        con_feat = con_feat * mask.unsqueeze(-1) # B x L x D
        
        return con_feat, mask
    
class SharedFC(nn.Module):
    def __init__(self, word_dim, hidden_dim, **kwargs):
        super(SharedFC, self).__init__()
        self.fc = torch.nn.Sequential(
            nn.Linear(word_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.out_dim = hidden_dim
        
    def forward(self, text, text_lengths):
        '''
        args
          text: B x L x wD
          text_lenghts: list of int (B)
        '''
        # Build mask
        B, L, _ = text.size()
        t_len = torch.tensor(text_lengths)
        mask = torch.arange(L).expand(B,L) < t_len.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=text.dtype, device=text.device) # B x L
        '''
        # Avg pool
        mask_ = mask.unsqueeze(-1) # B x L x 1
        avg_out = text * mask_ # B x L x wD
        avg_out = avg_out.sum(1) / mask_.sum(1).clamp(min=1) # B x wD
        avg_out = avg_out.unsqueeze(1).expand(-1,L,-1) # B x L x wD
        fc_out = torch.cat([avg_out,text], 2) 
        '''
        fc_out = text
        fc_feat = self.fc(fc_out) 
        fc_feat = fc_feat * mask.unsqueeze(-1) # B x L x D
        
        return fc_feat, mask

class SharedGRU(nn.Module):
    def __init__(self, word_dim, hidden_dim, **kwargs):
        super(SharedGRU, self).__init__()
        self.rnn = nn.GRU(word_dim, 512, batch_first=True, bidirectional=True)
        self.reduce = torch.nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.out_dim = hidden_dim 

    def forward(self, text, text_lengths):
        # Build mask
        B, L, _ = text.size()
        t_len = torch.tensor(text_lengths)
        mask = torch.arange(L).expand(B,L) < t_len.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=text.dtype, device=text.device) # B x L
        # GRU
        packed = pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        gru_init_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_out = padded[0] # B x L x 2*D
        gru_out = self.reduce(gru_out) # B x L x D
        gru_out = gru_out * mask.unsqueeze(-1)
    
        return gru_out, mask

    
def masked_softmax(inp, mask):
    '''
    args
      inp: B x L
      mask: B x L (1: value, 0: mask)
    return 
      att: B x L (softmax in -1 axis)
    '''
    mask = 1 - mask
    inp.data.masked_fill_(mask.data.bool(), -float("inf"))
    att = F.softmax(inp, dim = -1)
    att.data.masked_fill_(att.data != att.data, 0)  # remove nan from softmax on -inf
    return att