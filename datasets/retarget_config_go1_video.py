import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR


VISUALIZE_RETARGETING = True

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_description/urdf/go1_re.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

# generate mocap data name
OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/datasets/mocap_motions_go1_video_gen".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

REF_POS_SCALE = 1
INIT_POS = np.array([0, 0, 0.32])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [6, 11, 16, 21]
SIM_HIP_JOINT_IDS = [2, 7, 12, 17]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])
#SIM_TOE_OFFSET_LOCAL = [
#    np.array([0.2, 0.05, 0.0]),
#    np.array([0.2, -0.05, 0.0]),
#    np.array([0, -0.2, 0.0]),
#    np.array([0, 0.2, 0.0])
#]
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, 0.06, 0.0]),
    np.array([0, -0.06, 0.0]),
    np.array([0, 0.03, 0.0]),
    np.array([0, -0.03, 0.0])
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
        "video0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/video/walk_joint_pos5_from_csv.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 1, 222, 1
    ],
]