import copy
import functools
import os

import blobfile as bf
import numpy as np
#import torch as th
# import torch.distributed as dist
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
# from torch.optim import AdamW
import jittor as jt
from jittor import nn
from jittor.optim import AdamW

import io

from diffuseq.utils import logger
from diffuseq.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from diffuseq.utils.nn import update_ema
from diffuseq.step_sample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
jt.flags.use_cuda = 1

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        learning_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=10,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping
        self.device = "cuda" if jt.has_cuda else "cpu"
        self.use_ddp = False
        self.step = 0
        self.resume_step = 0
        self.ddp_model = self.model
        self.model_params = list(self.ddp_model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = jt.has_cuda

        self.checkpoint_path = checkpoint_path # DEBUG **

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if jt.has_cuda: # DEBUG **
            self.model.cuda()
            
        else:
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
       
        if (os.path.exists(resume_checkpoint)):
            print("Resumed")
            if resume_checkpoint[-3:] == '.pt':
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    jt.load(
                        actual_model_path(resume_checkpoint)
                    )
                )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = jt.load(
                actual_model_path(ema_checkpoint)
            )
            ema_params = self._state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if bf.exists(main_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {main_checkpoint}")
            state_dict = jt.load(
                actual_model_path(main_checkpoint)
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.learning_steps
            or self.step + self.resume_step < self.learning_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                # pass
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                batch_eval, cond_eval = next(self.eval_data)
                self.forward_only(batch_eval, cond_eval)
                print('eval on validation set')
                logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_only(self, batch, cond):
        with jt.no_grad():
            # self.ddp_model.zero_grad()
            # zero_grad(self.model_params,self.opt)
            self.opt.zero_grad()
            print("MicroBatchSize: ",self.microbatch)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch]
                micro_cond = {
                    k: v[i: i + self.microbatch]
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v*weights for k, v in losses.items()}
                )


    def forward_backward(self, batch, cond):
        # zero_grad(self.model_params,self.opt)
        self.opt.zero_grad()
        # self.ddp_model.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            # print(micro_cond.keys())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"]
                )
            # print(weights.dtype)
            loss = (losses["loss"] * weights).mean()
            # for k, v in losses.items():
            #     print(f"Value: {k}")
            #     print(f"Value: ",v)
            log_loss_dict(
                self.diffusion, t, {k: v*weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                self.opt.backward(loss * loss_scale)
                # (loss * loss_scale).backward()
            else:
                self.opt.backward(loss)

    def optimize_fp16(self):
        if any(not jt.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm=self.gradient_clipping #3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            jt.nn.utils.clip_grad_norm_(
                self.model.parameters(), #amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )
    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        self.master_params = list(self.ddp_model.parameters())
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        # Update ema
        self.master_params = list(self.ddp_model.parameters())
        new_emas = []
        for rate, params in zip(self.ema_rate, self.ema_params):
            updated = []
            for targ, src in zip(params, self.master_params):
                updated.append(targ*rate + src*(1-rate))
            new_emas.append(updated)
        self.ema_params = new_emas
        


    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            # print(cnt, p) ## DEBUG
            # print(cnt, p.grad)
            # cnt += 1
            grad = p.opt_grad(self.opt)
            if grad is not None:
                sqsum += (grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = (self.step + self.resume_step) / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            print('writing to', bf.join(get_blob_logdir(), filename))
            print('writing to', bf.join(self.checkpoint_path, filename))
            # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
            #     th.save(state_dict, f)
            with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f: # DEBUG **
                jt.save(state_dict, f.name) # save locally
                    # pass # save empty

        # save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)



    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        # print(state_dict)
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename[-3:] == '.pt':
        return int(filename[-9:-3])
    else:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            # print(f"{key}_q{quartile}", sub_loss)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
    # print("end")/

def actual_model_path(model_path):
    return model_path