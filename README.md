conda create -n i2p-env python=3.10

conda activate i2p-env

===

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

or

conda install pytorch torchvision torchaudio cpuonly -c pytorch

===

conda install ipykernel scipy matplotlib


Then download the data into the data folder:
- data
   - depths
   - images
   - mocap
   - points_colors.ply
   - points.ply
   - rotate_poses.py
   - transforms.json