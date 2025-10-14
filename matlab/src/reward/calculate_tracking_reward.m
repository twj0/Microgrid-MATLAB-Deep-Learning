function trackingReward = calculate_tracking_reward(action_W, net_load_W)
%CALCULATE_TRACKING_REWARD Net-load tracking reward term.
%   trackingReward = CALCULATE_TRACKING_REWARD(action_W, net_load_W)
%   computes a quadratic penalty on the mismatch between the battery power
%   command and the local net load. Perfect tracking (Action = -NetLoad)
%   yields zero penalty. No external configuration is required; tune the
%   gamma constant below if necessary.
%
%   Inputs:
%       action_W   - Storage dispatch command [W], >0 charge, <0 discharge.
%       net_load_W - Local net load [W], positive when load exceeds PV.
%
%   Output:
%       trackingReward - Scalar tracking reward contribution (<= 0).

    arguments
        action_W (1,1) double
        net_load_W (1,1) double
    end

    gamma = 1e-8;                     % Quadratic penalty coefficient.
    mismatch_W = action_W + net_load_W;
    trackingReward = -gamma * (mismatch_W ^ 2);
end
