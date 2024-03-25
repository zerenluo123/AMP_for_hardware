# ! this env is for training navigation policy for fixed locomotion

from time import time
import numpy as np
import os
import pygame
from termcolor import cprint

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .aliengo_nav_config import AliengoNavCfg

from legged_gym.envs.base.legged_robot import euler_from_quaternion


class AliengoNav(LeggedRobot):
    cfg: AliengoNavCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # allocate buffers
        # ! for locomotion policy
        self.num_locomotion_obs = self.cfg.env.num_locomotion_observations
        self.locomotion_obs_buf = torch.zeros(self.num_envs, self.num_locomotion_obs, device=self.device, dtype=torch.float)

        # ! load pre-trained locomotion policy
        locomotion_policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                              self.cfg.env.locomotion_policy_experiment_name,
                                              self.cfg.env.locomotion_policy_load_run,
                                              'exported_s1', 'actor.pt')
        cprint(f'Loading locomotion policy from \n {locomotion_policy_path}', 'cyan', attrs=['bold'])
        self.locomotion_policy = torch.jit.load(locomotion_policy_path).to(self.device)
        cprint(f'Loaded locomotion policy \n {self.locomotion_policy}', 'cyan', attrs=['bold'])

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_dict, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return self.obs_dict

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        roll_cutoff = torch.abs(self.roll) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals # already reach the last goal, that start the goal over again
        height_cutoff = self.root_states[:, 2] < -0.25

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= height_cutoff

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

    def _get_nav_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:11] = 0. # deta yaw and next delta yaw
        noise_vec[11:23] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[23:35] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[35:37] = 0. # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()

        # ! for locomotion policy
        self.locomotion_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.locomotion_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel

        # ! for navigation observation's noise
        self.nav_noise_scale_vec = self._get_nav_noise_scale_vec(self.cfg)


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): velocity command. Tensor of shape (num_envs, num_actions_per_env)
        """
        # ! action now is the 2-dimensional high-level command -- vx, yaw
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # action_lower = torch.Tensor([[-self.cfg.normalization.clip_action_lin_vel_x,
        #                               -self.cfg.normalization.clip_action_ang_vel_yaw]]).to(self.device)
        # print("action_lower", action_lower)
        # action_upper = torch.Tensor([[self.cfg.normalization.clip_action_lin_vel_x,
        #                               self.cfg.normalization.clip_action_ang_vel_yaw]]).to(self.device)
        # print("action_upper", action_upper)
        # self.action = torch.clip(actions, action_lower, action_upper).to(self.device)
        # print("before clip ", actions)
        # print("after clip ", self.actions)


        # ! calculate / concatenate the locomotion observation
        self.locomotion_commands[:, 0] = self.actions[:, 0] # vx
        self.locomotion_commands[:, 1] = 0                  # vy
        self.locomotion_commands[:, 2] = self.actions[:, 1] # yaw vel
        self.locomotion_obs_buf = torch.cat(( self.base_ang_vel  * self.obs_scales.ang_vel,
                                            self.projected_gravity,
                                            self.locomotion_commands * self.commands_scale,
                                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                            self.dof_vel * self.obs_scales.dof_vel,
                                            self.locomotion_actions
                                            ),dim=-1)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # ! feed the locomotion observation into the locomotion policy, output the action of locomotion policy
            self.locomotion_actions = self.locomotion_policy(self.locomotion_obs_buf)

            # ! still get torque via locomotion actions, still set dof actuation force tensor
            self.torques = self._compute_torques(self.locomotion_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # assign to obs struct
        self.obs_dict['obs'] = self.obs_buf
        self.obs_dict['privileged_obs'] = self.privileged_obs_buf.to(self.device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras, reset_env_ids

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
                    calls self._post_physics_step_callback() for common computations
                    calls self._draw_debug_vis() if needed
                """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # ! euler angle
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_goals()

        return env_ids

    def compute_observations(self):
        """
            Computes observations
        """
        # body orientation
        self.delta_yaw = self.target_yaw - self.yaw
        self.delta_next_yaw = self.next_target_yaw - self.yaw

        # navigation observation
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.delta_yaw[:, None],
                                    self.delta_next_yaw[:, None],
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add other privileged information
        if self.cfg.privileged_info.enable_foot_contact:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, contact), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        # if self.num_obs == self.num_privileged_obs - 6:
        #     self.obs_buf = self.privileged_obs_buf[:, 6:]
        # if self.num_obs == self.num_privileged_obs - 3:
        #     self.obs_buf = self.privileged_obs_buf[:, 3:]
        # else:
        #     self.obs_buf = torch.clone(self.privileged_obs_buf)
        self.obs_buf = self.privileged_obs_buf[:, 3:3+34]

        # ! add proprioceptive observation history
        # get previous step's obs
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()

        # get current step's obs
        cur_obs_buf = self.obs_buf.clone().unsqueeze(1)

        # concatenate to get full history
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        # Note: if reset, then the history buffer are all filled with the current observation
        at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, :] = self.obs_buf[at_reset_env_ids].unsqueeze(1)

        self.proprio_hist_buf = self.obs_buf_lag_history[:, -self.include_history_steps:].clone()




