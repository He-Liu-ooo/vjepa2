# python -m evals.main \
#   --val_only \
#   --fname configs/inference/vitg-384/ssv2.yaml \
#   --folder /home/hel19/workspace/repos/neural_network/vjepa2/output \
#   --override_config_folder \
#   --debugmode True \
#   --checkpoint /home/hel19/workspace/dataset/V-JEPA-2/ssv2/vitg-384/latest.pt \
#   --devices cuda:0

# python -m notebooks.vjepa2_demo

python notebooks/energy_landscape_example.py