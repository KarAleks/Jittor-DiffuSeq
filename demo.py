
import argparse
import os, json
from tracemalloc import start

import numpy as np
import jittor as jt

from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
from dpm_solver_jittor import NoiseScheduleVP, model_wrapper, DPM_Solver

import time
from diffuseq.utils import dist_util,logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)


import gradio as gr
import sys
jt.flags.use_cuda = 1
def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, rejection_rate=0.0, note='none')
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False, start_n=0)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# def main():
args = create_argparser().parse_args()
# dist_util.setup_dist()
logger.configure()

# load configurations.
config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
print(config_path)
# sys.setdefaultencoding('utf-8')
with open(config_path, 'rb', ) as f:
    training_args = json.load(f)
training_args['batch_size'] = 1
args.__dict__.update(training_args)

logger.log("### Creating model and diffusion...")

print('#'*10, args.clamp_step)
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, load_defaults_config().keys())
)

model.load_state_dict(
    dist_util.load_state_dict(args.model_path)
)

pytorch_total_params = sum(p.numel() for p in model.parameters())
logger.log(f'### The parameter count is {pytorch_total_params}')

# model.to(dist_util.dev())
model.eval()

tokenizer = load_tokenizer(args)
model_emb, tokenizer = load_model_emb(args, tokenizer)

model_emb.weight = model.word_embedding.weight.clone()
jt.save(model_emb.state_dict(),"asd.pt")
model_emb_copy = get_weights(model_emb, args)

set_seed(args.seed2)
start_t = time.time()

# batch, cond = next(data_valid)
# print(batch.shape)

SOLVER_STEP = 10

model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
# if not os.path.isdir(out_path):
#     os.mkdir(out_path)
#     out_path = os.path.join(out_path, f"seed{args.seed2}_solverstep{SOLVER_STEP}_{args.note}.json")
#     # fout = open(out_path, 'a')

    
def generate_text(data_input,chat_history):  
    print(data_input)
    all_test_data = []  
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split="demo",
        loaded_vocab=tokenizer,
        model_emb=model_emb,
        loop=False,
        demo_text = data_input
    )
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=jt.array(diffusion.betas))

    model_kwargs = {}
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="x_start",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="uncond",
    )


    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')
    for cond in all_test_data:
        
        input_ids_x = cond.pop('input_ids')
        print(input_ids_x)
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = jt.randn_like(x_start)
        input_ids_mask = jt.broadcast(input_ids_mask.unsqueeze(dim=-1), x_start.shape)
        x_noised = jt.where(input_ids_mask==0, x_start, noise)
        
        x_sample = dpm_solver.sample(
            x_noised,
            steps=SOLVER_STEP,
            order=2,
            skip_type="time_uniform",
            method="multistep",
            input_ids_mask=input_ids_mask,
            x_start=x_start,
        )

        all_sentence = [x_sample.cpu().numpy()]
        word_lst_recover = []


        arr = np.concatenate(all_sentence, axis=0)
        x_t = jt.array(arr)

        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        cands = jt.topk(logits, k=1, dim=-1)
        # sample = cands.indices

        for seq, input_mask in zip(cands[1], input_ids_mask_ori):
            # print(input_mask)
            len_x = args.seq_len - sum(input_mask)
            tokens = tokenizer.decode_token(seq[len_x:])
            tokens = tokens.replace("[CLS]","")
            tokens = tokens.replace("[PAD]","")
            tokens = tokens.replace("[SEP]","")
            word_lst_recover.append(tokens)
        chat_history.append((data_input,word_lst_recover[0]))
        return word_lst_recover[0]


if __name__ == "__main__":

    interface = gr.ChatInterface(
        fn=generate_text,       
        type="messages",   
        title="Jittor DiffuSeq Chat",
        description="A simple Gradio app for testing DiffuSeq models",
    )
    interface.launch()