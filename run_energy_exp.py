# run_energy_exp.py
import argparse
import os
import torch
import numpy as np
import random
from exp.energy_exp import Exp_Energy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='时序预测大模型在能源场景中的应用')
    
    # 随机种子
    parser.add_argument('--random_seed', type=int, default=2025, help='随机种子')
    
    # 基本配置
    parser.add_argument('--is_training', type=int, default=1, help='是否训练模式')
    parser.add_argument('--model_id', type=str, default='energy', help='模型ID')
    parser.add_argument('--model', type=str, default='EnergyPatchTST', help='模型名称')
    
    # 数据加载
    parser.add_argument('--data', type=str, default='wind_power', help='数据集名称')
    parser.add_argument('--root_path', type=str, default='./data/energy/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='wind_power.csv', help='数据文件')
    parser.add_argument('--features', type=str, default='M', help='预测任务类型(M/S/MS)')
    parser.add_argument('--target', type=str, default='power', help='目标特征')
    parser.add_argument('--freq', type=str, default='h', help='时间编码频率')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点位置')
    
    # 预测任务
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='开始标记长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    
    # 模型特殊功能
    parser.add_argument('--use_energy_data', type=bool, default=False, help='使用能源特定数据加载器')
    parser.add_argument('--use_future_vars', type=bool, default=False, help='使用未来已知变量')
    parser.add_argument('--future_vars', type=str, nargs='+', default=['temperature', 'wind_speed'], help='未来已知变量列表')
    parser.add_argument('--future_var_dim', type=int, default=10, help='未来变量维度')
    parser.add_argument('--use_multi_scale', type=bool, default=True, help='使用多尺度特征')
    parser.add_argument('--scale_levels', type=int, default=3, help='尺度级别数')
    parser.add_argument('--use_uncertainty', type=bool, default=True, help='使用不确定性估计')
    parser.add_argument('--n_samples', type=int, default=10, help='Monte Carlo采样数')
    parser.add_argument('--mc_dropout_rate', type=float, default=0.1, help='MC Dropout率')
    
    # 预训练和微调
    parser.add_argument('--use_pretrain', type=bool, default=True, help='使用预训练')
    parser.add_argument('--pretrain_datasets', type=str, nargs='+', default=['ETTh1', 'ETTh2'], help='预训练数据集')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='预训练轮数')
    parser.add_argument('--pretrain_lr', type=float, default=5e-4, help='预训练学习率')
    parser.add_argument('--pretrain_patience', type=int, default=5, help='预训练早停耐心')
    parser.add_argument('--finetune_epochs', type=int, default=3, help='微调轮数')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='微调学习率')
    parser.add_argument('--finetune_patience', type=int, default=3, help='微调早停耐心')
    parser.add_argument('--use_lr_scheduler', type=bool, default=True, help='使用学习率调度器')
    
    # PatchTST参数
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='全连接丢弃率')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='头部丢弃率')
    parser.add_argument('--patch_len', type=int, default=16, help='补丁长度')
    parser.add_argument('--stride', type=int, default=8, help='步长')
    parser.add_argument('--padding_patch', default='end', help='补丁填充方式')
    parser.add_argument('--revin', type=int, default=1, help='RevIN')
    parser.add_argument('--affine', type=int, default=0, help='RevIN仿射')
    parser.add_argument('--subtract_last', type=int, default=0, help='减去最后')
    parser.add_argument('--decomposition', type=int, default=0, help='分解')
    parser.add_argument('--kernel_size', type=int, default=25, help='分解核大小')
    parser.add_argument('--individual', type=int, default=0, help='独立头')
    
    # 模型配置
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入大小')
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入大小')
    parser.add_argument('--c_out', type=int, default=7, help='输出大小')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.05, help='丢弃率')
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='是否输出注意力')
    
    # 优化
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载器工作线程数')
    parser.add_argument('--itr', type=int, default=1, help='实验次数')
    parser.add_argument('--train_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--des', type=str, default='energy', help='实验描述')
    parser.add_argument('--loss', type=str, default='mse', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type3', help='学习率调整')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='多GPU设备ID')
    
    args = parser.parse_args()
    
    # 固定随机种子
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # GPU设置
    if torch.cuda.is_available() and args.use_gpu:
        args.use_gpu = True
        torch.cuda.set_device(args.gpu)
    else:
        args.use_gpu = False
    
    # 多GPU设置
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('实验参数:')
    print(args)
    
    # 创建实验
    Exp = Exp_Energy
    
    if args.is_training:
        for ii in range(args.itr):
            # 设置实验记录
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.des, 
                ii
            )
            
            exp = Exp(args)
            print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.des, 
            ii
        )
        
        exp = Exp(args)
        print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()