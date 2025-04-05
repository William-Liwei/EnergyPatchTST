# utils/pretrain_finetune.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.tools import EarlyStopping

class PretrainFinetune:
    """预训练和微调管理器"""
    
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.model.to(self.device)
    
    def pretrain(self, datasets, model_path=None):
        """
        在多个数据集上预训练模型
        datasets: 字典，包含多个数据集的数据加载器
        model_path: 保存预训练模型的路径
        """
        # 预训练参数
        epochs = self.args.pretrain_epochs if hasattr(self.args, 'pretrain_epochs') else 50
        patience = self.args.pretrain_patience if hasattr(self.args, 'pretrain_patience') else 10
        lr = self.args.pretrain_lr if hasattr(self.args, 'pretrain_lr') else 1e-4
        batch_size = self.args.batch_size
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 学习率调度器
        if hasattr(self.args, 'use_lr_scheduler') and self.args.use_lr_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr / 10)
        else:
            scheduler = None
        
        # 早停
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # Get checkpoint directory from model_path or create a default one
        if model_path:
            checkpoint_dir = os.path.dirname(model_path)
        else:
            checkpoint_dir = './checkpoints/pretrain'
        
        # Make sure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            # 遍历所有数据集
            for dataset_name, dataset in datasets.items():
                train_loader = dataset['train']
                val_loader = dataset.get('val')
                
                # 训练
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # 准备输入
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    # 预测
                    outputs = self._get_model_output(batch_x, batch_x_mark)
                    
                    # 计算损失
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = self._compute_loss(outputs, batch_y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                # 验证
                if val_loader is not None:
                    val_loss = self._validate(val_loader)
                    
                    # 早停检查
                    early_stopping(val_loss, self.model, checkpoint_dir)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step()
            
            # 输出训练状态
            avg_epoch_loss = epoch_loss / n_batches
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
            
            # 检查早停
            if early_stopping.early_stop:
                break
        
        # 保存预训练模型
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            print(f"Pretrained model saved to {model_path}")
        
        return self.model
    
    def finetune(self, train_loader, val_loader=None, model_path=None):
        """
        在特定数据集上微调模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        model_path: 保存微调模型的路径
        """
        # 微调参数
        epochs = self.args.finetune_epochs if hasattr(self.args, 'finetune_epochs') else 20
        patience = self.args.finetune_patience if hasattr(self.args, 'finetune_patience') else 5
        lr = self.args.finetune_lr if hasattr(self.args, 'finetune_lr') else 5e-5
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 学习率调度器
        if hasattr(self.args, 'use_lr_scheduler') and self.args.use_lr_scheduler:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.3
            )
        else:
            scheduler = None
        
        # 早停
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # Create checkpoint directory for early stopping
        if model_path:
            checkpoint_dir = os.path.dirname(model_path)
        else:
            checkpoint_dir = './checkpoints/finetune'
        
        # Make sure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练循环
        best_val_loss = float('inf')
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            # 训练
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 准备输入
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 预测
                outputs = self._get_model_output(batch_x, batch_x_mark)
                
                # 计算损失
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = self._compute_loss(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # 验证
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if model_path:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        torch.save(self.model.state_dict(), model_path)
                        print(f"Best model saved to {model_path}")
                
                # 早停检查 - This line needs to be updated
                early_stopping(val_loss, self.model, checkpoint_dir)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 输出训练状态
            avg_epoch_loss = epoch_loss / n_batches
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
            
            # 检查早停
            if early_stopping.early_stop:
                break
        
        # 如果没有验证集，直接保存最终模型
        if val_loader is None and model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            print(f"Final model saved to {model_path}")
        
        return self.model
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        n_samples = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                # 准备输入
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 预测
                outputs = self._get_model_output(batch_x, batch_x_mark)
                
                # 计算损失
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = self._compute_loss(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)
        
        self.model.train()
        return total_loss / n_samples
    
    def _get_model_output(self, batch_x, batch_x_mark=None, future_vars=None):
        """Get model output, handling different model architectures"""
        
        # Determine if this is an EnergyPatchTST model or standard PatchTST
        is_energy_model = 'EnergyPatchTST' in self.model.__class__.__name__
        
        if hasattr(self.model, 'use_uncertainty') and self.model.use_uncertainty:
            # Handle uncertainty models
            try:
                mean, var = self.model(batch_x, batch_x_mark, future_vars)
                return mean
            except TypeError:
                # Fall back to simpler calls if needed
                try:
                    mean, var = self.model(batch_x, batch_x_mark)
                    return mean
                except TypeError:
                    mean, var = self.model(batch_x)
                    return mean
        else:
            # Handle standard models with different interfaces
            try:
                if is_energy_model:
                    # Try EnergyPatchTST-specific call
                    return self.model(batch_x, batch_x_mark, future_vars)
                else:
                    # Try standard PatchTST call
                    return self.model(batch_x)
            except TypeError:
                # Fallback approaches
                try:
                    return self.model(batch_x, batch_x_mark)
                except TypeError:
                    return self.model(batch_x)
    
    def _compute_loss(self, outputs, targets):
        """计算损失"""
        # 基本均方误差损失
        criterion = nn.MSELoss()
        
        # 如果使用不确定性估计
        if hasattr(self.model, 'use_uncertainty') and self.model.use_uncertainty and isinstance(outputs, tuple):
            mean, var = outputs
            # 负对数似然损失
            loss = torch.mean(torch.log(var) / 2 + (targets - mean)**2 / (2 * var))
            return loss
        else:
            return criterion(outputs, targets)