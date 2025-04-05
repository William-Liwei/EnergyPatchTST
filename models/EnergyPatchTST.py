# models/EnergyPatchTST.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from models.PatchTST import Model as PatchTST_Model

class Model(nn.Module):
    def __init__(self, configs, max_seq_len=1024, d_k=None, d_v=None, norm='BatchNorm', attn_dropout=0.,
                act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True,
                pre_norm=False, store_attn=False, pe='zeros', learn_pe=True, pretrain_head=False, head_type='flatten', verbose=False, **kwargs):
        
        super().__init__()
        
        # 基础配置
        self.configs = configs
        self.task_name = configs.task_name if hasattr(configs, 'task_name') else 'forecasting'
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out if hasattr(configs, 'c_out') else configs.enc_in
        
        # 模型类型配置
        self.decomposition = configs.decomposition == 1
        self.kernel_size = configs.kernel_size if hasattr(configs, 'kernel_size') else 25
        self.individual = configs.individual == 1
        self.channels_last = True
        
        # 特殊功能开关
        self.use_future_vars = hasattr(configs, 'use_future_vars') and configs.use_future_vars
        self.use_multi_scale = hasattr(configs, 'use_multi_scale') and configs.use_multi_scale
        self.use_uncertainty = hasattr(configs, 'use_uncertainty') and configs.use_uncertainty
        
        # 多尺度参数
        if self.use_multi_scale:
            self.scale_levels = configs.scale_levels if hasattr(configs, 'scale_levels') else 3
            self.scale_win_sizes = [1, 7, 30]  # 小时级、日级、月级
        
        # 不确定性估计参数
        if self.use_uncertainty:
            self.n_samples = configs.n_samples if hasattr(configs, 'n_samples') else 10
            self.mc_dropout_rate = configs.mc_dropout_rate if hasattr(configs, 'mc_dropout_rate') else 0.1
        
        # 初始化骨干网络
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            
            # 趋势分量模型
            self.model_trend = self._build_backbone(
                d_k=d_k, d_v=d_v, norm=norm, attn_dropout=attn_dropout,
                act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, max_seq_len=max_seq_len,
                pretrain_head=pretrain_head, head_type=head_type, verbose=verbose
            )
            
            # 残差分量模型
            self.model_res = self._build_backbone(
                d_k=d_k, d_v=d_v, norm=norm, attn_dropout=attn_dropout,
                act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, max_seq_len=max_seq_len,
                pretrain_head=pretrain_head, head_type=head_type, verbose=verbose
            )
        else:
            self.model = self._build_backbone(
                d_k=d_k, d_v=d_v, norm=norm, attn_dropout=attn_dropout,
                act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, max_seq_len=max_seq_len,
                pretrain_head=pretrain_head, head_type=head_type, verbose=verbose
            )
        
        # 多尺度特征融合层
        if self.use_multi_scale:
            self.scale_projections = nn.ModuleList([
                nn.Linear(configs.d_model, configs.d_model // self.scale_levels)
                for _ in range(self.scale_levels)
            ])
            self.scale_fusion = nn.Linear(configs.d_model, configs.d_model)
        
        # 未来已知变量处理层
        if self.use_future_vars:
            self.future_var_dim = configs.future_var_dim if hasattr(configs, 'future_var_dim') else 10
            self.future_var_proj = nn.Linear(self.future_var_dim, configs.d_model)
            self.fusion_layer = nn.Linear(configs.d_model * 2, configs.d_model)
        
        # 不确定性估计层
        if self.use_uncertainty:
            self.mc_dropout = nn.Dropout(self.mc_dropout_rate)
            self.aleatoric_head = nn.Linear(configs.d_model, self.c_out * 2)  # 均值和方差
    
    def _build_backbone(self, d_k, d_v, norm, attn_dropout, act, key_padding_mask, 
                       padding_var, attn_mask, res_attention, pre_norm, store_attn, 
                       pe, learn_pe, max_seq_len, pretrain_head, head_type, verbose):
        """构建PatchTST主干网络"""
        return PatchTST_backbone(
            c_in=self.enc_in, 
            context_window=self.seq_len, 
            target_window=self.pred_len, 
            patch_len=self.configs.patch_len, 
            stride=self.configs.stride,
            max_seq_len=max_seq_len, 
            n_layers=self.configs.e_layers, 
            d_model=self.configs.d_model,
            n_heads=self.configs.n_heads, 
            d_k=d_k, 
            d_v=d_v, 
            d_ff=self.configs.d_ff, 
            norm=norm, 
            attn_dropout=attn_dropout,
            dropout=self.configs.dropout, 
            act=act, 
            key_padding_mask=key_padding_mask, 
            padding_var=padding_var,
            attn_mask=attn_mask, 
            res_attention=res_attention, 
            pre_norm=pre_norm, 
            store_attn=store_attn,
            pe=pe, 
            learn_pe=learn_pe, 
            fc_dropout=self.configs.fc_dropout, 
            head_dropout=self.configs.head_dropout, 
            padding_patch=self.configs.padding_patch,
            pretrain_head=pretrain_head, 
            head_type=head_type, 
            individual=self.individual, 
            revin=self.configs.revin == 1, 
            affine=self.configs.affine == 1,
            subtract_last=self.configs.subtract_last == 1,
            verbose=verbose
        )
    
    def _process_multi_scale(self, x):
        """处理多尺度特征"""
        batch_size, seq_len, n_vars = x.shape
        scales = []
        
        # 处理不同尺度
        for i, win_size in enumerate(self.scale_win_sizes):
            if win_size == 1:  # 原始尺度
                scale_x = x
            else:  # 聚合尺度
                padding_size = (win_size - (seq_len % win_size)) % win_size
                padded_x = F.pad(x, (0, 0, 0, padding_size))
                padded_len = seq_len + padding_size
                # 重塑并聚合
                reshaped_x = padded_x.reshape(batch_size, padded_len // win_size, win_size, n_vars)
                scale_x = torch.mean(reshaped_x, dim=2)  # 平均池化
            
            # 投影到相同维度
            if self.channels_last:
                scale_x = scale_x.permute(0, 2, 1)  # [B, C, L]
            
            scales.append(scale_x)
        
        return scales
    
    def _process_future_vars(self, future_vars, x_enc):
        """处理未来已知变量"""
        # 投影未来变量
        future_emb = self.future_var_proj(future_vars)
        
        # 将未来变量信息与encoder输出融合
        if self.channels_last:
            future_emb = future_emb.permute(0, 2, 1)  # [B, C, L]
        
        # 融合
        combined = torch.cat([x_enc, future_emb], dim=1)
        fused = self.fusion_layer(combined)
        
        return fused
    
    def forward(self, x, x_mark=None, future_vars=None):
        """
        前向传播
        x: [Batch, Input_len, Channel]
        x_mark: 时间特征标记 [Batch, Input_len, Mark_dim]
        future_vars: 未来已知变量 [Batch, Pred_len, Future_dim]
        """
        # 标准化输入
        if x_mark is not None and not self.channels_last:
            x = torch.cat([x, x_mark], dim=-1)
        
        # 多尺度处理
        if self.use_multi_scale:
            scales = self._process_multi_scale(x)
            multi_scale_outputs = []
            
            for i, scale_x in enumerate(scales):
                # 通过骨干网络处理每个尺度
                if self.decomposition:
                    res_init, trend_init = self.decomp_module(scale_x)
                    if self.channels_last:
                        res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
                    res = self.model_res(res_init)
                    trend = self.model_trend(trend_init)
                    out = res + trend
                else:
                    if self.channels_last:
                        scale_x = scale_x.permute(0, 2, 1)  # [B, Channel, Length]
                    out = self.model(scale_x)
                
                # 投影
                out = self.scale_projections[i](out)
                multi_scale_outputs.append(out)
            
            # 融合多尺度输出
            x_enc = torch.cat(multi_scale_outputs, dim=1)
            x_enc = self.scale_fusion(x_enc)
        else:
            # 标准处理
            if self.decomposition:
                res_init, trend_init = self.decomp_module(x)
                if self.channels_last:
                    res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
                res = self.model_res(res_init)
                trend = self.model_trend(trend_init)
                x_enc = res + trend
            else:
                if self.channels_last:
                    x = x.permute(0, 2, 1)  # [B, Channel, Length]
                x_enc = self.model(x)
        
        # 处理未来已知变量（如果有）
        if self.use_future_vars and future_vars is not None:
            x_enc = self._process_future_vars(future_vars, x_enc)
        
        # 不确定性估计
        if self.use_uncertainty and self.training:
            # 训练时使用Monte Carlo Dropout
            mc_samples = []
            for _ in range(self.n_samples):
                mc_out = self.mc_dropout(x_enc)
                mc_out = self.aleatoric_head(mc_out)
                mc_samples.append(mc_out)
            
            # 计算均值和方差
            mc_samples = torch.stack(mc_samples, dim=0)
            mean = torch.mean(mc_samples[:, :, :, :self.c_out], dim=0)
            var = torch.mean(mc_samples[:, :, :, self.c_out:], dim=0) + torch.var(mc_samples[:, :, :, :self.c_out], dim=0)
            
            # 返回均值和方差
            if self.channels_last:
                mean = mean.permute(0, 2, 1)
                var = var.permute(0, 2, 1)
            return mean, var
        elif self.use_uncertainty and not self.training:
            # 测试时直接预测均值和方差
            output = self.aleatoric_head(x_enc)
            mean, var = output[:, :, :self.c_out], output[:, :, self.c_out:]
            
            if self.channels_last:
                mean = mean.permute(0, 2, 1)
                var = var.permute(0, 2, 1)
            return mean, var
        else:
            # 标准输出
            if self.channels_last:
                x_enc = x_enc.permute(0, 2, 1)  # [B, Length, Channel]
            return x_enc