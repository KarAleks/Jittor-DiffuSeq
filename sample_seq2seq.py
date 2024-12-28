"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import jittor as jt
from jittor import nn
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)


def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@jt.no_grad()
def main():
    args = create_argparser().parse_args()
    logger.configure()

    # world_size = dist.get_world_size() or 1
    rank = 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(jt.load(args.model_path))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False)

    tokenizer = load_tokenizer(args)
    model_emb = nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim
    )
    model_emb.weight.assign(model.word_embedding.weight.clone().numpy())
    model_emb.eval().requires_grad_(False)
    jt.save(model_emb.state_dict(),"scjdbhc")
    jt.set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb,  # using the same embedding wight with tranining data
        loop=False
    )
    jt.flags.use_cuda = 1
    start_t = time.time()
    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')
    print("Clamp step: ",args.clamp_step)
    all_test_data = []

    idx = 0

    try:
        while  True:
            # print("akcbsdh")
            batch, cond = next(data_valid)
            # print(batch.shape)
            # if idx % world_size == rank:  # Split data per nodes/
            all_test_data.append(cond)
            # idx += 1
    except StopIteration:
        print('### End of reading iteration...')

    # model_emb.to()

    # if idx % world_size and rank >= idx % world_size:
    #     all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    for cond in iterator:
        # print("jvblsvb12")

        input_ids_x = jt.array(cond.pop('input_ids')).to(np.int32)
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = jt.array(cond.pop('input_mask')).to(np.int32)
        input_ids_mask_ori = input_ids_mask
        # print(input_ids_mask)
        noise = jt.randn_like(x_start)
        input_ids_mask = jt.broadcast(input_ids_mask.unsqueeze(dim=-1), x_start.shape)
        x_noised = jt.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}
        
        if args.step == args.diffusion_steps:
            # print("es a chishty")
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # for input_mask in input_ids_mask_ori:
        #     # input_mask = input_mask.to(np.int32)
        #     print(sum(input_mask),input_mask.sum())
        #     len_x = args.seq_len - sum(input_mask).tolist()
        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
        # print("shdvsfvbfvjkhs")
        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )
        
        # print(samples[0].shape) # samples for each step

        sample = samples[-1]

        # print('decoding for seq2seq', )
        # print(sample.shape)

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = jt.topk(logits, k=1, dim=-1)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        # tokenizer = load_tokenizer(args)
        input_ids_mask_ori = input_ids_mask_ori.to(np.int32)
        for seq, input_mask in zip(cands[1], input_ids_mask_ori):
            # input_mask = input_mask.to(np.int32)
            len_x = args.seq_len - sum(input_mask).item()
            # print(len_x)
            # print(len(seq[len_x:].view(-1).tolist()))
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = args.seq_len - sum(input_mask).item()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        # for i in range(world_size):
            # if i == rank:  # Write files sequentially
        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
            print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
        fout.close()
        

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
