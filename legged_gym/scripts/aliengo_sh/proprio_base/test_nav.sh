#!/bin/bash
CACHE=$1
python legged_gym/scripts/test_env_nav.py --task=aliengo_nav \
--runner_class_name=ProprioBaseAMPOnPolicyRunner \
--locomotion_policy_experiment_name=aliengo_amp_example \
--locomotion_policy_load_run="${CACHE}" \
--locomotion_policy_checkpoint=-1


