#!/bin/bash
CACHE=$1
python legged_gym/scripts/play.py --task=aliengo_amp \
--load_run="${CACHE}" --checkpoint=-1 \
--runner_class_name=ProprioBaseAMPOnPolicyRunner \
--export_policy --export_onnx_policy \
--locomotion_policy_experiment_name=aliengo_amp_example \
--locomotion_policy_load_run="${CACHE}" \
--locomotion_policy_checkpoint=-1

