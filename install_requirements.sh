#!/bin/bash
#torch
#ogb
#dgl
#sortedcontainers
#pyvis
#pynvml
#tqdm
#pymetis
# """Setup  pip  install for cuda11.1 or cuda 11.2."""
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter
pip install torch-sparse
pip install torch-geometric
pip install https://data.dgl.ai/wheels/dgl_cu111-0.9.1-cp36-cp36m-manylinux1_x86_64.whl
pip install tqdm
pip install ogb
pip install pynvml
sudo pip install matplotlib
pip install pyvis
pip install tabulate
pip install sortedcontainers
pip install torch-summary
pip install pymetis
pip install seaborn

# """Setup  pip  install for cuda10.1."""
# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.7.0%2Bcu101/torch_scatter-2.0.6-cp36-cp36m-linux_x86_64.whl
# pip install torch-sparse
# pip install torch-geometric
# # if use "pip install dgl-cu101 -f https://data.dgl.ai/wheels/repo.html "
# # it will install the newest version of dgl
# # dgl 0.8.0 is not match with pytorch 1.7.1 
# # the dgl 0.7.2 can be compatible with pytorch 1.7.1
# pip install https://data.dgl.ai/wheels/dgl_cu101-0.7.2-cp36-cp36m-manylinux1_x86_64.whl

# pip install tqdm
# pip install ogb
# pip install pynvml
# sudo pip install matplotlib
# pip install pyvis
# pip install tabulate
# pip install sortedcontainers
# pip install torch-summary
# pip install pymetis
# pip install seaborn
