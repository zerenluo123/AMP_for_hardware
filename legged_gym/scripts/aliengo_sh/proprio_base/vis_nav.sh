#!/bin/bash
NAV_CACHE=$1
LOCO_CACHE=$2
python legged_gym/scripts/play_nav.py --task=aliengo_nav \
--load_run="${NAV_CACHE}" --checkpoint=-1 \
--runner_class_name=ProprioBaseNavOnPolicyRunner \
--locomotion_policy_experiment_name=aliengo_amp_example \
--locomotion_policy_load_run="${LOCO_CACHE}" \
--locomotion_policy_checkpoint=-1 \
--export_policy --export_onnx_policy


