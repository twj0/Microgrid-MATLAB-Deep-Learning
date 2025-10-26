function batch_plot_all_algorithms(varargin)
% batch_plot_all_algorithms 批量扫描 SAC/PPO/DDPG/DQN 的最新训练结果并生成可视化图片
%
% 功能：
% - 自动定位四个算法目录（matlab/DRL/{sac,ppo,ddpg,dqn}）
% - 在各算法目录下递归查找最新的 results*.mat（或包含 results 变量的 .mat 文件）
% - 对每个算法调用 plot_lr_vs_reward() 绘制图表，并统一保存到 project_document/figures/
%
% 用法示例：
%   % 使用默认参数：两子图、不保存（仅显示）
%   batch_plot_all_algorithms();
%
%   % 直接保存所有图像，启用移动平均奖励与对数学习率
%   batch_plot_all_algorithms('save_fig', true, 'ma_window', 10, 'log_lr', true);
%
% Name-Value 参数（会被透传给 plot_lr_vs_reward）：
%   'dual_axis'      - 默认 false
%   'save_fig'       - 默认 true（批量工具默认保存图片以便归档）
%   'output_dir'     - 默认 'project_document/figures/'
%   'ma_window'      - 默认 0（关闭移动平均）
%   'log_lr'         - 默认 false
%   'stage_positions'- 默认 []
%   'stage_labels'   - 默认 {}
%
% 作者：Augment Agent（自动生成）
% -------------------------------------------------------------------------

% ------------------------- 参数解析 --------------------------------------
p = inputParser; p.FunctionName = mfilename;
addParameter(p, 'dual_axis',       false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'save_fig',        true,  @(x) islogical(x) || isnumeric(x));
addParameter(p, 'output_dir',      "project_document/figures/", @(x) (ischar(x) || isstring(x)) && ~isempty(x));
addParameter(p, 'ma_window',       0,     @(x) isnumeric(x) && isscalar(x) && x>=0);
addParameter(p, 'log_lr',          false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'stage_positions', [],    @(x) isnumeric(x));
addParameter(p, 'stage_labels',    {},    @(x) iscell(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results; if isstring(opt.stage_labels), opt.stage_labels = cellstr(opt.stage_labels); end

% ------------------------- 路径准备 --------------------------------------
thisFile = mfilename('fullpath'); srcDir = fileparts(thisFile);
matlabDir = fileparts(srcDir);
drlRoot = fullfile(matlabDir, 'DRL');
algos = {'sac','ppo','ddpg','dqn'};

if ~exist(opt.output_dir, 'dir')
    try, mkdir(opt.output_dir); catch, end
end

fprintf('>>> 批量可视化开始：输出目录 = %s\n', char(opt.output_dir));

% ------------------------- 主循环 ----------------------------------------
for i = 1:numel(algos)
    alg = algos{i};
    algDir = fullfile(drlRoot, alg);
    if ~exist(algDir, 'dir')
        fprintf(' - [%s] 目录不存在：%s，跳过。\n', upper(alg), algDir);
        continue;
    end

    % 查找最新 results*.mat 或包含 results 变量的 .mat
    matFile = find_latest_results_mat(algDir);
    if isempty(matFile)
        fprintf(' - [%s] 未找到包含 results 的 MAT 文件，跳过。\n', upper(alg));
        continue;
    end

    fprintf(' - [%s] 加载：%s\n', upper(alg), matFile);
    try
        results = [];
        S = load(matFile, 'results');
        if isfield(S, 'results') && ~isempty(S.results) && isstruct(S.results)
            results = S.results;
        else
            % 兜底：完整加载并尝试从变量池中寻找（大小写不敏感）
            S = load(matFile);
            if isfield(S, 'results')
                results = S.results;
            else
                varNames = fieldnames(S);
                tf = find(strcmpi(varNames, 'results'), 1);
                if ~isempty(tf)
                    results = S.(varNames{tf});
                end
            end
        end
        if isempty(results)
            fprintf('   ⚠ 未能从 %s 中解析出 results 结构，跳过。\n', matFile);
            continue;
        end
    catch ME
        fprintf('   ⚠ 载入失败：%s（跳过）\n', ME.message);
        continue;
    end

    % 调用绘图函数
    try
        plot_lr_vs_reward(results, ...
            'dual_axis', opt.dual_axis, ...
            'save_fig',  opt.save_fig, ...
            'output_dir', opt.output_dir, ...
            'ma_window', opt.ma_window, ...
            'log_lr',    opt.log_lr, ...
            'stage_positions', opt.stage_positions, ...
            'stage_labels',    opt.stage_labels);
        drawnow;
        fprintf('   ✓ [%s] 绘图完成。\n', upper(alg));
    catch ME
        fprintf('   ⚠ 绘图失败：%s\n', ME.message);
    end
end

fprintf('>>> 批量可视化结束。\n');

end % function

% ============================== 子函数 ====================================
function matFile = find_latest_results_mat(rootDir)
% 在 rootDir 下递归查找包含 results 变量的 MAT 文件；优先匹配文件名含 "result"
matFile = '';
try
    L = dir(fullfile(rootDir, '**', '*.mat'));
    if isempty(L), return; end
    % 优先 results 命名
    names = lower(string({L.name}));
    score = double(contains(names, 'result')); % 1/0
    % 按 score（优先） + datenum（新->旧）排序
    [~, idx] = sortrows([score(:), [L.datenum]'], [-1 -1]);
    for k = idx(:)'
        f = fullfile(L(k).folder, L(k).name);
        try
            w = whos('-file', f);
            if any(strcmp({w.name}, 'results'))
                matFile = f; return;
            end
        catch
        end
    end
catch
end
end

