## Setup:
The code is based on Jittor.
```bash 
pip install -r requirements.txt 
```

## Datasets
Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use two datasets for our reimplementation.

| Task | Datasets | Training Samples | Source | Used in __*Jittor-DiffuSeq*__
|-|-|-|-|-|
| Open-domain Dialogue | Commonsense Conversation | 3382k | [CCM](https://github.com/thu-coai/ccm) | [download](https://drive.google.com/drive/folders/1exENF9Qc5UtXnHlNl9fvaxP3zyyH32qp?usp=sharing) |
| Question Generation | Quasar-T | 117k | [OpenQA](https://github.com/thunlp/OpenQA) | [download](https://drive.google.com/drive/folders/122YK0IElSnGZbPMigXrduTVL1geB4wEW?usp=sharing) |

## Jittor-DiffuSeq Training
```bash
cd scripts
bash train.sh
```
Arguments explanation:
- ```--dataset```: the name of datasets, just for notation
- ```--data_dir```: the path to the saved datasets folder, containing ```train.jsonl,test.jsonl,valid.jsonl```
- ```--seq_len```: the max length of sequence $z$ ($x\oplus y$)
- ```--resume_checkpoint```: if not none, restore this checkpoint and continue training
- ```--vocab```: the tokenizer is initialized using bert or load your own preprocessed vocab dictionary (e.g. using BPE)


## DiffuSeq Decoding
You need to modify the path to ```model_dir```, which is obtained in the training stage.
```bash
cd scripts
bash run_decode.sh
```

## Speed-up Decoding
Customize implementation of [DPM-Solver++](https://github.com/LuChengTHU/dpm-solver) for DiffuSeq to accelerate its sampling speed.
```bash
cd scripts
bash run_decode_solver.sh
```

## Evaluation
You need to specify the folder of decoded texts. This folder should contain the decoded files from the same model but sampling with different random seeds. 
```bash
cd scripts
python3 eval_seq2seq.py --folder ../{your-path-to-outputs} --mbr
```
Note: if you want to use this evaluation script for output files from other models, please make sure the same line from these output files refers to the same piece of data. Otherwise the diversity score could be incorrect.

## Trained Models
Trained models can be found [here](https://drive.google.com/drive/folders/1EnTEgiUUsSKE4NZHwZ5aPAaAZVqOZEAX?usp=sharing)

## Demo 
For running demo chat, train model or use pretrained models, you just need to specify model path
```bash
python3 demo.py --model_path {path-to-your-model-path}
```
## Citation
Please add the citation if DiffuSeq paper or code helps you.

```
@inproceedings{gong2022diffuseq,
  author = {Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  booktitle = {International Conference on Learning Representations, ICLR},
  title = {{DiffuSeq}: Sequence to Sequence Text Generation with Diffusion Models},
  year = 2023
}

@article{gong2023diffuseqv2,
  title={DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models},
  author={Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2310.05793},
  year={2023}
}

```
## Original Repo
[DiffuSeq Repo](https://github.com/Shark-NLP/DiffuSeq)
