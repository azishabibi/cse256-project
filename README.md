This is the code for UCSD cse 256 final project. It involves finetuning a quantized LoRA model (LoftQ) with McEval-Instruct dataset and evaluating with Evalplus dataset.


## Neccessary packages
Run ```pip install -r requirements.txt```.
Install Evalplus by ```pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"```
## Train the model
The script is used in AutoDL platform so I do not know if it works on other platforms. Make neccessary changes.
For master node:
```
export NCCL_SOCKET_IFNAME=eth1  # Replace eth1 with your network interface name, use ifconfig to check.
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=55568 \
    new_ddp.py
```
For worker node:
```
export NCCL_SOCKET_IFNAME=eth1 
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=1 \
    --master_addr="127.0.0.1" \
    --master_port=55568 \
    new_ddp.py
```
## Evaluate the model
Run the ```eval.ipynb```.
## launch the model locally
Run ```python run.py```