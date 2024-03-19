# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AliengoAMPCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 5480
        include_history_steps = 4  # Number of steps of history to include.
        num_observations = 45
        num_privileged_obs = 48
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        get_commands_from_joystick = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': -0.0,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'FR_hip_joint': 0.0,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RL_hip_joint': -0.0,   # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'RR_hip_joint': 0.0,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_aliengo_2rd_f100_it4000_ly2_mlp_dec27_dec28.pt"

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        measure_heights = False

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.25, 1.75]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = True  # actuator net: True
        added_lag_timesteps = 4

        randomize_Motor_Offset = True  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5


    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 2.4] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class privileged_info( LeggedRobotCfg.privileged_info ):
        enable_motor_strength = False
        enable_disturbance_force = False
        enable_foot_contact = False
        enable_foot_height = False

class AliengoAMPCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'ProprioBaseAMPOnPolicyRunner'

    class policy( LeggedRobotCfgPPO.policy ):
        class hist_encoder:
            include_history_steps = AliengoAMPCfg.env.include_history_steps
            priv_mlp_units = [258, 128, 3]

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'aliengo_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 25000 # number of policy updates

        amp_reward_coef = 2.0
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.2
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.01, 0.01, 0.01] * 4

        load_run = -1

        checkpoint_model = None # load pre-trained model name

        export_policy = False
        export_onnx_policy = False


