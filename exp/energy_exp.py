# exp/energy_exp.py
from exp.exp_main import Exp_Main
from utils.pretrain_finetune import PretrainFinetune
from data_provider.energy_data_loader import energy_data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, EnergyPatchTST
import torch
from torch import optim
import os
import numpy as np
import time
from utils.metrics import metric

class Exp_Energy(Exp_Main):
    """能源场景实验类"""
    
    def __init__(self, args):
        super(Exp_Energy, self).__init__(args)
        self.use_future_vars = hasattr(args, 'use_future_vars') and args.use_future_vars
        self.use_pretrain = hasattr(args, 'use_pretrain') and args.use_pretrain
    
    def _build_model(self):
        """构建模型"""
        model_dict = {
            'EnergyPatchTST': EnergyPatchTST,
            # 其他可能的能源场景模型...
        }
        
        if self.args.model in model_dict.keys():
            model = model_dict[self.args.model].Model(self.args).float()
        else:
            # 回退到原始模型构建
            return super(Exp_Energy, self)._build_model()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
    
    def _get_data(self, flag):
        """获取数据"""
        if hasattr(self.args, 'use_energy_data') and self.args.use_energy_data:
            # 使用能源特定数据加载器
            data_set, data_loader = energy_data_provider(self.args, flag)
        else:
            # 回退到原始数据加载
            data_set, data_loader = super(Exp_Energy, self)._get_data(flag)
        
        return data_set, data_loader
    
    def _process_model_input(self, batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars=None):
        """处理模型输入"""
        # 将数据转移到设备
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        if future_vars is not None:
            future_vars = future_vars.float().to(self.device)
        
        # 创建decoder输入
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, future_vars
    
    def _get_model_output(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, future_vars=None):
        """获取模型输出"""
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                
                # 使用我们的定制模型
                if self.use_future_vars and future_vars is not None:
                    outputs = self.model(batch_x, future_vars)
                else:
                    outputs = self.model(batch_x)
        else:

                # 使用我们的定制模型
                if self.use_future_vars and future_vars is not None:
                    outputs = self.model(batch_x, future_vars)
                else:
                    outputs = self.model(batch_x)
        
        return outputs
    
    def pretrain(self, setting):
        """预训练模型"""
        if not self.use_pretrain:
            print("Pretraining is disabled. Skipping...")
            return self.model
        
        # 创建预训练数据集集合
        pretrain_datasets = {}
        
        # 加载预训练数据集
        for dataset in self.args.pretrain_datasets:
            self.args.data = dataset
            self.args.data_path = f"{dataset}.csv"
            
            train_data, train_loader = self._get_data(flag='train')
            val_data, val_loader = self._get_data(flag='val')
            
            pretrain_datasets[dataset] = {
                'train': train_loader,
                'val': val_loader
            }
        
        # 创建预训练管理器
        pretrainer = PretrainFinetune(self.model, self.device, self.args)
        
        # 预训练路径
        pretrain_path = os.path.join(self.args.checkpoints, setting, 'pretrain')
        os.makedirs(pretrain_path, exist_ok=True)
        model_path = os.path.join(pretrain_path, 'pretrained_model.pth')
        
        # 执行预训练
        pretrainer.pretrain(pretrain_datasets, model_path)
        
        return self.model
    
    def train(self, setting):
        """训练模型"""
        # 预训练（如果启用）
        if self.use_pretrain:
            self.pretrain(setting)
        
        # 加载数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 提取批次数据
                if self.use_future_vars and len(batch) > 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    future_vars = None
                
                # 处理输入
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, future_vars = self._process_model_input(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars)
                
                # 获取输出
                outputs = self._get_model_output(batch_x, batch_x_mark, dec_inp, batch_y_mark, future_vars)
                
                # 不确定性处理
                if hasattr(self.model, 'use_uncertainty') and self.model.use_uncertainty and isinstance(outputs, tuple):
                    outputs, uncertainty = outputs
                
                # 计算损失
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # 验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        """验证模型"""
        total_loss = []
        uncertainty_values = []
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                # 提取批次数据
                if self.use_future_vars and len(batch) > 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    future_vars = None
                
                # 处理输入
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, future_vars = self._process_model_input(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars)
                
                # 获取输出
                outputs = self._get_model_output(batch_x, batch_x_mark, dec_inp, batch_y_mark, future_vars)
                
                # 不确定性处理
                if hasattr(self.model, 'use_uncertainty') and self.model.use_uncertainty and isinstance(outputs, tuple):
                    outputs, uncertainty = outputs
                    uncertainty_values.append(uncertainty.mean().item())
                
                # 计算损失
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)
                
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        
        # 输出不确定性信息（如果有）
        if uncertainty_values:
            avg_uncertainty = np.mean(uncertainty_values)
            print(f"Average prediction uncertainty: {avg_uncertainty:.4f}")
        
        return total_loss
    
    def test(self, setting, test=0):
        """测试模型"""
        # 与原始方法相同，但支持future_vars和uncertainty
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        preds = []
        trues = []
        uncertainties = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # 提取批次数据
                if self.use_future_vars and len(batch) > 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    future_vars = None
                
                # 处理输入
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, future_vars = self._process_model_input(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_vars)
                
                # 获取输出
                outputs = self._get_model_output(batch_x, batch_x_mark, dec_inp, batch_y_mark, future_vars)
                
                # 不确定性处理
                if hasattr(self.model, 'use_uncertainty') and self.model.use_uncertainty and isinstance(outputs, tuple):
                    outputs, uncertainty = outputs
                    uncertainties.append(uncertainty.detach().cpu().numpy())
                
                # 处理输出
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                pred = outputs
                true = batch_y
                
                preds.append(pred)
                trues.append(true)
                
                # 可视化样本
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    
                    # 如果有不确定性，也可视化
                    if uncertainties:
                        unc = uncertainties[-1][0, :, -1]
                        plt.figure()
                        plt.plot(range(len(pd)), pd, 'b', label='Prediction')
                        plt.fill_between(
                            range(len(pd)), 
                            pd - 1.96 * np.sqrt(unc), 
                            pd + 1.96 * np.sqrt(unc), 
                            color='blue', 
                            alpha=0.2, 
                            label='95% Confidence'
                        )
                        plt.plot(range(len(gt)), gt, 'r', label='Ground Truth')
                        plt.legend()
                        plt.savefig(os.path.join(folder_path, str(i) + '_uncertainty.pdf'))
                        plt.close()
        
        # 计算指标
        preds = np.array(preds)
        trues = np.array(trues)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # 如果有不确定性，也处理
        if uncertainties:
            uncertainties = np.array(uncertainties)
            uncertainties = uncertainties.reshape(-1, uncertainties.shape[-2], uncertainties.shape[-1])
        
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        
        # 保存到结果文件
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
        
        # 保存预测结果
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        # 保存不确定性结果（如果有）
        if uncertainties:
            np.save(folder_path + 'uncertainty.npy', uncertainties)
        
        return