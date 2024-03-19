import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR


VISUALIZE_RETARGETING = True

# URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_description/urdf/go1_re.urdf".format(
#     LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo_re.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
# generate mocap data name
OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/datasets/mocap_motions_aliengo_video_gen_x0.4".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

REF_POS_SCALE = 0.94  #10----1   18-5 -----0.8
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
#         10        #
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, 0.11, 0.0]), # FR
    np.array([0, -0.10, 0.0]),
    np.array([0, 0.06, 0.0]),
    np.array([0, 0.05, 0.0])
]
# #        18-5         #
# SIM_TOE_OFFSET_LOCAL = [
#     np.array([0, -0.0, 0.0]),
#     np.array([0, 0.0, 0.0]),
#     np.array([0, 0.03, 0.0]),
#     np.array([0, -0.03, 0.0])
# ]
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
        "video00",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/video/video10runandjump_changethin_from_csv_x0.4.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 20, 300, 1
    ],
    [
        "video00",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/video/video10runandjump_changethin_from_csv_x0.4.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 20, 300, 1
    ],
    [
        "video00",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/video/video10runandjump_changethin_from_csv_x0.4.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 20, 300, 1
    ],
    [
        "video00",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/video/video10runandjump_changethin_from_csv_x0.4.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 20, 300, 1
    ],
]
