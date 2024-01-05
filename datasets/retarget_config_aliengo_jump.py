import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR


VISUALIZE_RETARGETING = True

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo_re.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/datasets/mocap_motions_aliengo_jumpL".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

REF_POS_SCALE = 0.925
INIT_POS = np.array([0, 0, 0.32])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [6, 11, 16, 21]
SIM_HIP_JOINT_IDS = [2, 7, 12, 17]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, 0.09, 0.0]),
    np.array([0, -0.09, 0.0]),
    np.array([0, 0.09, 0.0]),
    np.array([0, -0.09, 0.0])
]
TOE_HEIGHT_OFFSET = 0.02

DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    [
        "right_turn1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 560, 675, 1.0
    ],
    [
        "right_turn2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk01_joint_pos_sym.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 378, 530, 1
    ],
    [
        "right_turn3",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos_sym.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 1223, 1454, 1.0
    ],
    [
        "left_turn1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos_sym.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 560, 675, 1.0
    ],
    [
        "left_turn2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk01_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 378, 530, 1
    ],
    [
        "left_turn3",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 1223, 1454, 1.0
    ],
    [
        "jump00",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_jump00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 979, 1032, 6.0
    ],
    [
        "jump01",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_stepup_jump0_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 590, 644, 1.0
    ],
    [
        "jump02",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_stepup_jump1_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 403, 463, 3.0
    ],
    [
        "jump03",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_stepup_jump2_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 21, 90, 1.0
    ],
    [
        "jump04",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_stepup_jump3_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 487, 545, 3.0
    ],
]