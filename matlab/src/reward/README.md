# Reward Module

## 设计理念
奖励体系遵循“净负荷自洽 + 机会主义套利”的基本策略：
- **功率平衡优先**：电池优先对冲本地源荷不匹配，维持微电网自洽运行。
- **经济效益辅助**：当价差信号强烈时，允许策略理性偏离平衡来赚取收益。
- **专家修正可扩展**：可选的模糊逻辑用于在极端工况下注入人工经验。

当前实现仅包含“跟踪 + 经济”两类核心奖励，电池健康项已移除，方便快速集成。需要时可在此目录下添加新的子奖励，并在混合函数中扩展权重组合逻辑。

## 文件说明
- `calculate_tracking_reward.m`：二次惩罚跟踪误差（`Action + NetLoad`），偏差越小惩罚越小。
- `calculate_economic_reward.m`：线性购售电收益 + 峰时购电二次惩罚，鼓励低价充电、高价放电。
- `calculate_hybrid_reward.m`：将上述子奖励按固定权重组合，必要时叠加模糊修正，输出最终奖励及分量。

## 调用接口
```matlab
[reward, components] = calculate_hybrid_reward( ...
    action_W, ...        % 储能功率指令，充电为正
    net_load_W, ...      % 净负荷（负载-光伏），正值代表缺电
    P_grid_W, ...        % 与电网功率交换，购电为正
    price_USD_per_kWh, ... % 当前电价
    soc_percent, ...     % 当前 SOC（模糊逻辑可用）
    fuzzy_system);       % (可选) 模糊控制器/回调，为空则禁用
```

- `reward` 为最终混合奖励。
- `components` 提供 `tracking`、`economic`、`fuzzy` 以及权重信息，便于调试与可视化。

## 参数调节方式
所有关键常数直接写在各自函数中：
- `calculate_tracking_reward`：`gamma` 控制跟踪惩罚斜率。
- `calculate_economic_reward`：峰值阈值、电价阈值、二次惩罚系数、时间步长。
- `calculate_hybrid_reward`：`w_tracking`、`w_economic` 决定子奖励权重。

如需调参，可在代码中直接修改对应常数并重新仿真（也可对接 Simulink Mask 变量或外部脚本进行覆盖）。

## 模糊修正（可选）
- 传入 `fuzzy_system` 后会自动调用。支持类型：`function_handle`、`mamfis/sugfis` 对象、任意带 `evalfis` 方法的对象或包含 `eval` 方法的结构体。
- 函数内部将输入向量构造成 `[SOC(%), Price($/kWh), NetLoad(kW)]`，其中净负荷会除以 1000 与默认 FIS 量纲匹配。
- 若模糊求值失败，函数会捕获异常并返回 0，同时给出警告。可按需改写为抛出异常以便调试。
- 示例 FIS 文件位于 `model/fuzzy_correction.fis`，可直接加载或作为自定义设计的参考模板。

## Simulink 集成步骤
1. 在 MATLAB Function 模块中调用 `calculate_hybrid_reward`，传入观测向量对应信号。
2. 将输出连至 RL Agent 或自定义控制器的奖励端口。
3. 若启用模糊逻辑，可在模型初始化脚本中加载 `.fis` 文件并把对象传入函数的第六个输入端口。

## 训练建议
1. **阶段一**：将 `w_economic` 调整为 0，只保留跟踪项，让智能体先学会平衡功率。
2. **阶段二**：逐步恢复经济项权重，并根据实验曲线调节峰值惩罚系数，实现稳定套利。
3. 记录 `components` 中各子奖励的时间序列，辅助分析策略在不同电价/负荷情景下的决策偏好。

---
后续若需要扩展其他奖励或约束，可按“独立函数 + 混合入口组合”的模式继续扩展，保持模块化结构，便于测试与维护。
