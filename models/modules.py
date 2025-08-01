import torch
import torch.nn as nn
import torch.nn.functional as F
# abbreviation: MMG: Missing Modality Generator, CAP: Context-Aware Prompter

# class MMG(nn.Module):
#     def __init__(self, dropout_rate, n, d):
#         super(MMG, self).__init__()
#         self.n = n
#         self.d = d
#         # self.W = nn.Parameter(torch.randn(n, d, dtype=torch.cfloat)) 
#         # self.W = nn.Parameter(torch.randn(1, d, dtype=torch.cfloat)) 
#         self.W_real = nn.Parameter(torch.randn(1, d))
#         self.W_imag = nn.Parameter(torch.randn(1, d))
#         #  MMG(n=max_text_len, d=hs, dropout_rate=dropout_rate)
#         self.layer_norm = nn.LayerNorm(d)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.linear = nn.Linear(d, d)

#     def forward(self, F_l):
#        # print(f"F_l size  {F_l.size()}")
#         F_l = torch.mean(F_l, dim=1)
#        # print(f"afterffl F_l size  {F_l.size()}")
#         X_l = torch.fft.fft(F_l, dim=1)  
#        # print(f"X_l size  {X_l.size()}")
#         Wc = torch.complex(self.W_real, self.W_imag)
#         X_tilde_l = Wc * X_l
#         F_tilde_l = torch.fft.ifft(X_tilde_l, dim=1).real
#         F_l = self.layer_norm(F_l + self.dropout(F_tilde_l))
#         F_l = self.linear(F_l)
#         return F_l


class MMG(nn.Module):
    def __init__(self, dropout_rate, n, d):
        super(MMG, self).__init__()
        self.n = n  # 序列长度n
        self.d = d  # 特征维度d
        self.W = nn.Parameter(torch.randn(n, d, dtype=torch.cfloat))  # 参数W的形状 [n, d]
        self.layer_norm = nn.LayerNorm(d)  # 层归一化在最后一个维度上进行
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.linear = nn.Linear(d, d)  # 线性变换层

    def forward(self, F_l):
        # 输入: F_l 形状 [batch, seqlen, dim]
        
        # 对序列维度(seqlen)求平均值
        F_l = torch.mean(F_l, dim=1)  # 输出形状 [batch, dim]

        # 进行傅里叶变换（注意：这里dim=1是因为现在只剩下两个维度）
        X_l = torch.fft.fft(F_l, dim=1)  # 输出形状 [batch, dim] (复数)

        # 将参数W扩展到与批次大小匹配并进行乘法操作
        # 注意：如果d != dim 或者 n != seqlen 在初始化时会出错
        W_expanded = self.W.unsqueeze(0).expand(F_l.size(0), -1, -1)  # 扩展W的形状 [batch, n, d]
        X_tilde_l = W_expanded * X_l.unsqueeze(1)  # 为了广播相乘，X_l需要调整为 [batch, 1, dim]
                                                  # 输出形状 [batch, n, dim] (复数)

        # 进行逆傅里叶变换（注意：这里dim=1是因为我们希望在n这个维度上进行逆变换）
        F_tilde_l = torch.fft.ifft(X_tilde_l, dim=1).real  # 输出形状 [batch, n, dim] (实数)

        # 将原始的F_l与经过变换后的F_tilde_l相加，并应用dropout和层归一化
        # 注意：F_l的形状是 [batch, dim] 而 F_tilde_l 的形状是 [batch, n, dim]
        # 因此我们需要将F_l重复n次以匹配F_tilde_l的形状
        F_l_repeated = F_l.unsqueeze(1).repeat(1, self.n, 1)  # 输出形状 [batch, n, dim]
        F_l = self.layer_norm(F_l_repeated + self.dropout(F_tilde_l))  # 输出形状 [batch, n, dim]

        # 最后通过线性层
        F_l = self.linear(F_l)  # 输出形状 [batch, n, d]

        return F_l  # 输出形状 [batch, n, d]

class CAP(nn.Module):
    def __init__(self, prompt_length,dim=768):
        super(CAP, self).__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, V, T, r_i, r_t):
        V_to_V = self.attention(V, r_i)
        T_to_T = self.attention(T, r_t)
        return T_to_T, V_to_V

    def attention(self, query, key_value):
        b, k, s = key_value.shape

        q = self.q_proj(query)
        k = self.k_proj(key_value)  
        v = self.v_proj(key_value) 

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, v)  

        output = output.mean(dim=1)
        output = output.unsqueeze(1)
        return output





# class CAP(nn.Module):
#     def __init__(self, prompt_length, dim=768):
#         super(CAP, self).__init__()
#         self.dim = dim
#         self.prompt_length = prompt_length
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         # Remove AdaptiveAvgPool2d since it's not suitable for our use case.
#         # Instead, we will handle the pooling or reshaping in a more flexible way.

#     def forward(self, V, T, r_i, r_t):
#         """
#         Generates text and image prompts using attention mechanism.
#         """
#         V_to_V = self.attention(V, r_t)
#         T_to_T = self.attention(T, r_i)
        
#         # Reshape to ensure the output has shape [B, P, D]
#         # Assuming P is the desired prompt length. If needed, you can adjust this part.
#         V_to_V = V_to_V.unsqueeze(1).expand(-1, self.prompt_length, -1)
#         T_to_T = T_to_T.unsqueeze(1).expand(-1, self.prompt_length, -1)
        
#         return T_to_T, V_to_V

#     def attention(self, query, key_value):
#         b, k, s = key_value.shape
        
#         q = self.q_proj(query)  # Shape: (B, S, D)
#         k = self.k_proj(key_value)  # Shape: (B, K, D)
#         v = self.v_proj(key_value)  # Shape: (B, K, D)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  # Shape: (B, S, K)
#         attn_probs = F.softmax(attn_scores, dim=-1)  # Softmax across keys
        
#         output = torch.matmul(attn_probs, v)  # Shape: (B, S, D)
        
#         # To generate prompts, we might want to pool or reshape the outputs.
#         # Here, we'll simply take the mean of the sequence dimension to get a single vector per batch.
#         # This may need adjustment based on your specific needs.
#         output = output.mean(dim=1)  # Shape: (B, D)
        
#         return output