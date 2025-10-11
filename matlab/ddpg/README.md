# DDPG微电网储能优化

## 简介

这是一个简化的DDPG (Deep Deterministic Policy Gradient) 强化学习实现,用于优化微电网储能系统的充放电策略。

## 文件说明

- **main.m**: 主程序,负责加载数据、配置环境、启动训练
- **train_model.m**: 训练函数,创建DDPG智能体并执行训练
- **README.md**: 使用说明文档

## 使用步骤

### 1. 准备数据

首先运行数据生成脚本:
```matlab
cd ../src
run('generate_data.m')
```

这将生成:
- 光伏出力曲线 (pv_power_profile)
- 负载功率曲线 (load_power_profile)
- 峰谷电价曲线 (price_profile_ts)
- 电池特性查找表 (OCV_LUT, R_int_LUT)

### 2. 运行训练

```matlab
cd ../ddpg
run('main.m')
```

### 3. 训练过程

训练过程包括:
1. 加载仿真数据
2. 配置Simulink模型 (Microgrid.slx)
3. 定义强化学习环境
4. 创建DDPG智能体
5. 执行训练 (50回合, 每回合30天)
6. 保存训练结果

## 技术参数

### 观测空间 (9维)
1. SOC - 电池荷电状态 (%)
2. PV_power - 光伏出力 (W)
3. Load_power - 负载功率 (W)
4. Price - 电价 (元/kWh)
5. Net_load - 净负荷 (W)
6. Battery_power - 电池功率 (W)
7. Grid_power - 电网功率 (W, 正值=从电网取电, 负值=向电网送电)
8. Reward - 奖励值
9. Time_of_day - 当前小时 (0-23)

### 动作空间 (1维)
- Battery_power - 电池充放电功率 (-100kW ~ +100kW)
  - 负值: 放电
  - 正值: 充电

### DDPG参数
- **Actor网络**: [128, 64] 隐藏层
- **Critic网络**: [128, 64] 隐藏层
- **学习率**: Actor=5e-4, Critic=1e-3
- **折扣因子**: 0.99
- **批量大小**: 64
- **经验缓冲区**: 1e6
- **采样时间**: 3600秒 (1小时)

### 训练参数
- **最大回合数**: 50
- **每回合步数**: 720 (30天 × 24小时)
- **目标奖励**: 500
- **平均窗口**: 5回合

## 输出结果

训练完成后会生成:
- `ddpg_agent_YYYYMMDD_HHMMSS.mat` - 保存的智能体和训练结果
- 训练进度曲线图

## 注意事项

1. **Simulink模型**: 确保 `model/Microgrid.slx` 存在且配置正确
2. **RL Agent模块**: 模型中需要有 `RL Agent` 模块
3. **数据路径**: 确保数据文件路径正确
4. **MATLAB版本**: 需要 Reinforcement Learning Toolbox
5. **仿真时长**: 默认30天训练周期,可根据需要调整

## 下一步

训练完成后可以:
1. 分析训练曲线,评估收敛性
2. 测试智能体在新场景下的表现
3. 调整网络结构或超参数
4. 延长训练周期以提升性能

## 故障排查

### 常见问题

**Q: 数据文件不存在**
- A: 先运行 `matlab/src/generate_data.m` 生成数据

**Q: 模型加载失败**
- A: 检查 `model/Microgrid.slx` 路径是否正确

**Q: 训练过程中断**
- A: 检查Simulink模型配置,确保能正常运行

**Q: 奖励值异常**
- A: 检查模糊推理系统 (FIS) 配置是否正确
