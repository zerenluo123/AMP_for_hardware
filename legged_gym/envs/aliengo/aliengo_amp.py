# ! this env is for training locomotion AMP policy

from time import time
import numpy as np
import os
import pygame

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .aliengo_amp_config import AliengoAMPCfg


class AliengoAMP(LeggedRobot):
    cfg: AliengoAMPCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._get_commands_from_joystick = self.cfg.env.get_commands_from_joystick
        if self._get_commands_from_joystick:
            pygame.init()
            self._p1 = pygame.joystick.Joystick(0)
            self._p1.init()
            print(f"Loaded joystick with {self._p1.get_numaxes()} axes.")


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def _init_buffers(self):
        super()._init_buffers()

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)

    def compute_observations(self):
        super().compute_observations()
        if self._get_commands_from_joystick:
          for event in pygame.event.get():
            lin_vel_x = -1 * self._p1.get_axis(1)
            if lin_vel_x >= 0:
             lin_vel_x *= torch.abs(torch.tensor(self.command_ranges["lin_vel_x"][1]))
            else:
             lin_vel_x *= torch.abs(torch.tensor(self.command_ranges["lin_vel_x"][0]))

            lin_vel_y = -1 * self._p1.get_axis(3)
            if lin_vel_y >= 0:
             lin_vel_y *= torch.abs(torch.tensor(self.command_ranges["lin_vel_y"][1]))
            else:
             lin_vel_y *= torch.abs(torch.tensor(self.command_ranges["lin_vel_y"][0]))

            ang_vel = -1 * self._p1.get_axis(0)
            if ang_vel >= 0:
             ang_vel *= torch.abs(torch.tensor(self.command_ranges["ang_vel_yaw"][1]))
            else:
             ang_vel *= torch.abs(torch.tensor(self.command_ranges["ang_vel_yaw"][0]))

            self.commands[:, 0] = lin_vel_x
            self.commands[:, 1] = lin_vel_y
            self.commands[:, 2] = ang_vel

        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
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
        self.obs_buf = self.privileged_obs_buf[:, 3:3+45]

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





