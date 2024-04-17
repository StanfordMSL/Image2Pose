conda create -n i2p-env python=3.10

conda activate i2p-env

===

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

or

conda install pytorch torchvision torchaudio cpuonly -c pytorch

===

conda install ipykernel scipy matplotlib
