import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5
       
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, T, S):
        q = self.linear_q(T)  # query
        k = self.linear_k(S)  # key
        v = self.linear_v(S)  # value
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale
        attn_weights = self.softmax(attn_weights)
        
        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        # 输出结果
        output = self.linear_out(attn_output)
        return output
    
class SelfAttention_For_Fusion_uncertainty(nn.Module):
    def __init__(self, hidden_dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        # self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.Softplus = nn.Softplus()

    def forward(self, feat_rgb, feat_tir, attn_rgb, attn_tir):
        """
        T: 模板特征 [B, N_query, C]
        feat_rgb: RGB 搜索区域特征 [B, N_key, C]
        feat_tir: TIR 搜索区域特征 [B, N_key, C]
        attn_rgb: 来自ViT的注意力 [B, H, N_query, N_key]
        attn_tir: 来自ViT的注意力 [B, H, N_query, N_key]
        """
        # === step1: 计算 evidence ===
        evidence_rgb = attn_rgb.mean(1)   # [B, Nq, N_key]
        evidence_tir = attn_tir.mean(1)   # [B, Nq, N_key]

        # print(evidence_rgb.size(), evidence_tir.size())

        alpha_rgb = self.Softplus(evidence_rgb) + 1
        alpha_tir = self.Softplus(evidence_tir) + 1

        S_rgb = torch.sum(alpha_rgb, dim=-1, keepdim=True)  # [B, Nq, 1]
        S_tir = torch.sum(alpha_tir, dim=-1, keepdim=True)

        u_rgb = feat_rgb.size(1) / S_rgb  # 不确定性
        u_tir = feat_tir.size(1) / S_tir

        # === step2: 归一化权重 (模态内) ===
        # w_rgb = alpha_rgb / S_rgb.unsqueeze(-1).expand(evidence_rgb.shape)
        # w_tir = alpha_tir / S_tir.unsqueeze(-1).expand(evidence_tir.shape)


        # === step3: min-max 归一化到 [0.01, 1]  tokens 权重===
        w_all = (alpha_rgb + alpha_tir) / 2
        min_v, max_v = w_all.min(), w_all.max()
        weights = 0.01 + (w_all - min_v) / (max_v - min_v + 1e-6) * (1 - 0.01)
        # weights = weights.mean(-1).unsqueeze(-1)
        # === step4: 融合不确定性权重 ===
        # u_rgb,u_tir = torch.softmax(torch.cat([u_rgb.unsqueeze(-1),u_tir.unsqueeze(-1)],dim=-1),dim=-1).chunk(2,dim=-1)
        u_rgb,u_tir = u_rgb.unsqueeze(-1),u_tir.unsqueeze(-1)
        u_fused =  2*u_rgb * u_tir / (u_rgb + u_tir + 1e-6)
        w_rgb_final = u_fused / u_rgb
        w_tir_final = u_fused / u_tir


        # print(feat_rgb.size(), feat_tir.size(), w_rgb_final.size(), w_tir_final.size(), weights.size())
        # === step5: 加权特征 ===
        # v_rgb = self.linear_v(feat_rgb* w_rgb_final.unsqueeze(-1)* weights.unsqueeze(-1))
        v_rgb = feat_rgb* w_rgb_final* weights.unsqueeze(-1)
        # v_rgb = self.proj_drop(v_rgb)

        # v_tir = self.linear_v(feat_tir * w_tir_final.unsqueeze(-1)* weights.unsqueeze(-1))
        v_tir = feat_tir * w_tir_final* weights.unsqueeze(-1)
        # v_tir = self.proj_drop(v_tir)
        v_f = v_rgb + v_tir
        # v_f = self.linear_out(v_rgb + v_tir)

        # v_f = torch.cat([v_f, v_f], dim=1)
        return v_f, [u_fused, u_rgb, u_tir], weights


class TemplateRouter(nn.Module):
    def __init__(self, L, C):
        super(TemplateRouter, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  
    
        self.fc = nn.Sequential(
            nn.Linear(C, 64),  
            nn.ReLU(),
            nn.Linear(64, 1)  
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, template1, template2):
        """
        :param template1: [B, L, C]
        :param template2: [B, L, C]
        :return: [B, 2] 评分张量，和相对强弱的归一化评分
        """
        template1 = template1.permute(0, 2, 1)  # [B, C, L]
        template2 = template2.permute(0, 2, 1)  # [B, C, L]

        template1 = self.gap(template1.unsqueeze(2))  
        template2 = self.gap(template2.unsqueeze(2))  
        

        template1 = template1.view(template1.size(0), -1) 
        template2 = template2.view(template2.size(0), -1) 

        score1 = self.fc(template1)  # [B, 1]
        score2 = self.fc(template2)  # [B, 1]
        
        scores = torch.cat([score1, score2], dim=1)  # [B, 2]
        
        relative_score = self.softmax(scores)  # [B, 2]
        
        return scores, relative_score