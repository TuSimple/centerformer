## Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint)

Our experiments are tested on the following environments:

- Python: 3.9.12
- PyTorch: 1.9.1
- CUDA: 11.1
- We use spconv 1.2.1 in our experiment.

### Installation 

```bash
# basic python libraries
conda create --name centerformer python=3.9
conda activate centerformer
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
git clone [this repo]
cd centerformer
pip install -r requirements.txt
sh setup.sh

# add CenterFormer to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERFORMER"
```

Most of the libaraies are the same as [CenterPoint](https://github.com/tianweiy/CenterPoint) except for the transformer part. If you run into any issues, you can also refer to their detailed instructions and search from the issues in their repo.