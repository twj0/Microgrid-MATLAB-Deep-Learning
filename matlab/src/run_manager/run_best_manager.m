function varargout = run_best_manager(mode, varargin)
%RUN_BEST_MANAGER 管理训练过程中的最优结果
%   管理SAC训练的最优智能体和仿真数据,确保只保留奖励最高的结果
%
% 用法:
%   run_best_manager('init') - 初始化最优结果目录
%   run_best_manager('save', agent, results, episodeData) - 保存新结果(如果更优)
%   [agent, data, meta] = run_best_manager('load') - 加载最优结果
%   meta = run_best_manager('query') - 查询当前最优记录
%   run_best_manager('cleanup') - 清理临时文件
%
% 输入:
%   mode - 操作模式: 'init', 'save', 'load', 'query', 'cleanup'
%   varargin - 根据mode不同而变化的参数
%
% 输出:
%   根据mode返回不同内容

    % 获取最优结果存储目录
    bestDir = get_best_run_directory();
    
    switch lower(mode)
        case 'init'
            initialize_best_run_directory(bestDir);
            
        case 'save'
            if nargin < 3
                error('save模式需要提供: agent, results, [episodeData]');
            end
            agent = varargin{1};
            results = varargin{2};
            episodeData = [];
            if nargin >= 4
                episodeData = varargin{3};
            end
            save_if_better(bestDir, agent, results, episodeData);
            
        case 'load'
            [varargout{1}, varargout{2}, varargout{3}] = load_best_run(bestDir);
            
        case 'query'
            varargout{1} = query_best_meta(bestDir);
            
        case 'cleanup'
            cleanup_temp_files(bestDir);
            
        otherwise
            error('未知操作模式: %s。支持的模式: init, save, load, query, cleanup', mode);
    end
end

%% ========================================================================
%% 核心功能函数
%% ========================================================================

function bestDir = get_best_run_directory()
    %GET_BEST_RUN_DIRECTORY 获取最优结果存储目录
    
    run_manager_dir = fileparts(mfilename('fullpath'));
    project_root = find_project_root(run_manager_dir);
    bestDir = fullfile(project_root, 'results', 'best_run');
end

function initialize_best_run_directory(bestDir)
    %INITIALIZE_BEST_RUN_DIRECTORY 初始化最优结果目录
    
    if ~exist(bestDir, 'dir')
        mkdir(bestDir);
        fprintf('✓ 创建最优结果目录: %s\n', bestDir);
    else
        fprintf('  最优结果目录已存在: %s\n', bestDir);
    end
    
    % 创建README说明文件
    readmePath = fullfile(bestDir, 'README.txt');
    if ~exist(readmePath, 'file')
        fid = fopen(readmePath, 'w');
        fprintf(fid, '=== SAC训练最优结果存储目录 ===\n\n');
        fprintf(fid, '本目录自动保存训练奖励最高的结果，包括:\n');
        fprintf(fid, '1. best_agent.mat - 最优SAC智能体\n');
        fprintf(fid, '2. best_episode_data.mat - 最优episode的仿真数据(SOC/SOH/Power)\n');
        fprintf(fid, '3. best_run_meta.mat - 元数据(奖励、时间戳等)\n\n');
        fprintf(fid, '使用 load_and_visualize_best() 加载并可视化最优结果\n');
        fprintf(fid, '创建时间: %s\n', datestr(now));
        fclose(fid);
    end
end

function save_if_better(bestDir, agent, results, episodeData)
    %SAVE_IF_BETTER 如果当前结果更优则保存
    
    metaPath = fullfile(bestDir, 'best_run_meta.mat');
    
    % 加载历史最佳
    prevBest = struct('reward', -inf, 'timestamp', '', 'episodes', 0);
    if isfile(metaPath)
        try
            prevBest = load(metaPath);
            fprintf('\n📊 历史最佳奖励: %.2f (训练于 %s)\n', ...
                prevBest.reward, prevBest.timestamp);
        catch ME
            fprintf('⚠ 加载历史记录失败: %s，将创建新记录\n', ME.message);
        end
    else
        fprintf('\n📊 首次训练，将保存本次结果\n');
    end
    
    % 获取当前最佳奖励
    currentBest = get_best_reward(results);
    
    fprintf('  本轮最佳奖励: %.2f\n', currentBest);
    fprintf('  历史最佳奖励: %.2f\n', prevBest.reward);
    
    % 比较并决定是否保存
    isFirstValidSave = ~isfinite(prevBest.reward);
    if isFirstValidSave || currentBest > prevBest.reward
        fprintf('\n🎉 刷新纪录! 保存新的最优结果...\n');
        
        % 1. 原子性保存智能体
        agentPath = fullfile(bestDir, 'best_agent.mat');
        tempAgentPath = [agentPath, '.tmp'];
        try
            save(tempAgentPath, 'agent', '-v7.3');
            movefile(tempAgentPath, agentPath, 'f');
            fprintf('  ✓ 已保存智能体: %s\n', agentPath);
        catch ME
            fprintf('  ✗ 智能体保存失败: %s\n', ME.message);
            if exist(tempAgentPath, 'file')
                delete(tempAgentPath);
            end
        end
        
        % 2. 保存episode数据(如果提供)
        if ~isempty(episodeData)
            dataPath = fullfile(bestDir, 'best_episode_data.mat');
            tempDataPath = [dataPath, '.tmp'];
            try
                save(tempDataPath, '-struct', 'episodeData', '-v7.3');
                movefile(tempDataPath, dataPath, 'f');
                fprintf('  ✓ 已保存episode数据: %s\n', dataPath);
            catch ME
                fprintf('  ✗ Episode数据保存失败: %s\n', ME.message);
                if exist(tempDataPath, 'file')
                    delete(tempDataPath);
                end
            end
        end
        
        % 3. 原子性保存元数据
        newMeta = struct();
        newMeta.reward = currentBest;
        newMeta.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        newMeta.episodes = get_total_episodes(results);
        newMeta.training_time = get_training_time(results);
        newMeta.average_reward = get_average_reward(results);
        newMeta.matlab_version = version;
        newMeta.results_summary = results;
        
        tempMetaPath = [metaPath, '.tmp'];
        try
            save(tempMetaPath, '-struct', 'newMeta', '-v7.3');
            movefile(tempMetaPath, metaPath, 'f');
            fprintf('  ✓ 已更新元数据\n');
        catch ME
            fprintf('  ✗ 元数据保存失败: %s\n', ME.message);
            if exist(tempMetaPath, 'file')
                delete(tempMetaPath);
            end
        end
        
        % 4. 备份旧记录(可选)
        if prevBest.reward > -inf
            backup_old_record(bestDir, prevBest);
        end
        
        fprintf('\n💾 所有文件已保存至: %s\n', bestDir);
        fprintf('   奖励提升: %.2f → %.2f (+%.2f)\n', ...
            prevBest.reward, currentBest, currentBest - prevBest.reward);
        
    else
        improvement_needed = prevBest.reward - currentBest;
        fprintf('\n⊘ 未刷新纪录，保留历史最优结果\n');
        if isfinite(improvement_needed)
            fprintf('   需要提升 %.2f 才能超越历史最佳\n', improvement_needed);
        else
            fprintf('   当前训练分数无效，已跳过保存\n');
        end
    end
end

function [agent, episodeData, meta] = load_best_run(bestDir)
    %LOAD_BEST_RUN 加载最优训练结果
    
    agent = [];
    episodeData = [];
    meta = struct();
    
    % 检查目录是否存在
    if ~exist(bestDir, 'dir')
        warning('最优结果目录不存在: %s', bestDir);
        return;
    end
    
    % 加载智能体
    agentPath = fullfile(bestDir, 'best_agent.mat');
    if isfile(agentPath)
        try
            agentData = load(agentPath);
            agent = agentData.agent;
            fprintf('✓ 已加载最优智能体\n');
        catch ME
            warning(ME.identifier, '加载智能体失败: %s', ME.message);
        end
    else
        warning('未找到最优智能体文件: %s', agentPath);
    end
    
    % 加载episode数据
    dataPath = fullfile(bestDir, 'best_episode_data.mat');
    if isfile(dataPath)
        try
            episodeData = load(dataPath);
            fprintf('✓ 已加载episode数据\n');
        catch ME
            warning(ME.identifier, '加载episode数据失败: %s', ME.message);
        end
    else
        fprintf('  未找到episode数据(首次运行可能尚未生成)\n');
    end
    
    % 加载元数据
    metaPath = fullfile(bestDir, 'best_run_meta.mat');
    if isfile(metaPath)
        try
            meta = load(metaPath);
            fprintf('✓ 已加载元数据\n');
            fprintf('  - 最佳奖励: %.2f\n', meta.reward);
            fprintf('  - 训练时间: %s\n', meta.timestamp);
            fprintf('  - 训练回合: %d\n', meta.episodes);
        catch ME
            warning(ME.identifier, '加载元数据失败: %s', ME.message);
        end
    else
        warning('未找到元数据文件: %s', metaPath);
    end
end

function meta = query_best_meta(bestDir)
    %QUERY_BEST_META 查询最优记录的元数据
    
    meta = struct('exists', false, 'reward', NaN, 'timestamp', '', 'episodes', 0);
    
    metaPath = fullfile(bestDir, 'best_run_meta.mat');
    if isfile(metaPath)
        try
            loaded = load(metaPath);
            meta.exists = true;
            meta.reward = loaded.reward;
            meta.timestamp = loaded.timestamp;
            meta.episodes = loaded.episodes;
        catch ME
            warning(ME.identifier, '查询元数据失败: %s', ME.message);
        end
    end
end

function cleanup_temp_files(bestDir)
    %CLEANUP_TEMP_FILES 清理临时文件
    
    if ~exist(bestDir, 'dir')
        return;
    end
    
    tempFiles = dir(fullfile(bestDir, '*.tmp'));
    if isempty(tempFiles)
        fprintf('  无需清理，没有临时文件\n');
        return;
    end
    
    fprintf('清理临时文件...\n');
    for i = 1:length(tempFiles)
        tempPath = fullfile(bestDir, tempFiles(i).name);
        try
            delete(tempPath);
            fprintf('  ✓ 已删除: %s\n', tempFiles(i).name);
        catch ME
            fprintf('  ✗ 删除失败: %s (%s)\n', tempFiles(i).name, ME.message);
        end
    end
end

function backup_old_record(bestDir, prevMeta)
    %BACKUP_OLD_RECORD 备份旧的最优记录(保留最近3个)
    
    historyDir = fullfile(bestDir, 'history');
    if ~exist(historyDir, 'dir')
        mkdir(historyDir);
    end
    
    % 生成备份文件名
    timestamp_safe = strrep(prevMeta.timestamp, ':', '-');
    timestamp_safe = strrep(timestamp_safe, ' ', '_');
    backupName = sprintf('backup_%.2f_%s.mat', prevMeta.reward, timestamp_safe);
    backupPath = fullfile(historyDir, backupName);
    
    % 保存备份
    try
        copyfile(fullfile(bestDir, 'best_run_meta.mat'), backupPath);
        fprintf('  📦 已备份旧记录: %s\n', backupName);
    catch
        % 备份失败不是致命错误，忽略
    end
    
    % 清理过旧的备份(只保留最近3个)
    backups = dir(fullfile(historyDir, 'backup_*.mat'));
    if length(backups) > 3
        [~, idx] = sort([backups.datenum], 'descend');
        for i = 4:length(backups)
            delete(fullfile(historyDir, backups(idx(i)).name));
        end
    end
end

%% ========================================================================
%% 辅助函数
%% ========================================================================

function reward = get_best_reward(results)
    %GET_BEST_REWARD 从results结构体提取最佳奖励
    
    if isfield(results, 'best_reward')
        reward = results.best_reward;
    elseif isfield(results, 'episode_rewards') && ~isempty(results.episode_rewards)
        reward = max(results.episode_rewards);
    else
        reward = NaN;
    end
    
    if isempty(reward) || ~isfinite(reward)
        reward = -inf;
    end
end

function episodes = get_total_episodes(results)
    %GET_TOTAL_EPISODES 获取总训练回合数
    
    if isfield(results, 'total_episodes')
        episodes = results.total_episodes;
    elseif isfield(results, 'episode_rewards')
        episodes = nnz(~isnan(results.episode_rewards));
    else
        episodes = 0;
    end
end

function time_sec = get_training_time(results)
    %GET_TRAINING_TIME 获取训练耗时(秒)
    
    if isfield(results, 'training_time')
        time_sec = results.training_time;
    else
        time_sec = 0;
    end
end

function avg_reward = get_average_reward(results)
    %GET_AVERAGE_REWARD 获取平均奖励
    
    if isfield(results, 'average_reward')
        avg_reward = results.average_reward;
    elseif isfield(results, 'episode_rewards')
        valid_rewards = results.episode_rewards(~isnan(results.episode_rewards));
        if ~isempty(valid_rewards)
            avg_reward = mean(valid_rewards);
        else
            avg_reward = NaN;
        end
    else
        avg_reward = NaN;
    end
end

function project_root = find_project_root(start_dir)
    project_root = start_dir;
    max_depth = 10;
    for i = 1:max_depth
        if exist(fullfile(project_root, 'matlab'), 'dir') && exist(fullfile(project_root, 'model'), 'dir')
            return;
        end
        parent_dir = fileparts(project_root);
        if isempty(parent_dir) || strcmp(parent_dir, project_root)
            break;
        end
        project_root = parent_dir;
    end
    error('RunManager:ProjectRootNotFound', '无法从路径%s定位项目根目录', start_dir);
end

