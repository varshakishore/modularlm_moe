from pathlib import Path
import random 
import os
import numpy as np
import spacy
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timeit
from datetime import datetime


from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from collections import defaultdict

import json
from tqdm.auto import tqdm

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, GPT2Model, GPT2LMHeadModel
from transformers import PreTrainedTokenizerBase, default_data_collator
from datasets import load_dataset
from model.custom_gpt import GPT2LMHeadModelMoE
from st_moe_pytorch import MoE

from accelerate import Accelerator
import wandb

import evaluation
from torch.optim import AdamW
import argparse

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm

def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{args.wandb_name}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir

def get_dataloader(dataset, tokenizer, batch_size, max_seq_len, mode='language_modeling', shuffle=False, moe=False):
    assert mode in {'language_modeling'}

    ner = NamedEntityRecognition()

    def get_routing(text, tokenized_text):
        # words, tags = ner(text)
        len_bos_tag = 13
        tagging = ner(text[len_bos_tag:]) # remove bos token because it messes with tagging
        tagging = torch.cat([torch.zeros(len_bos_tag), tagging])
        routing = torch.zeros(tokenized_text['input_ids'].shape[-1]).to(tokenized_text['labels'].dtype)
        for i in range(1, tokenized_text['input_ids'].shape[-1]):
            span = tokenized_text.token_to_chars(i)
            if span is not None:
                #get majority tag in span
                tags = tagging[span.start:span.end]
                routing[i] = (tags.sum()>(len(tags)/2)).int() # majority vore in binary string
        return routing
    
    def tokenization(example):
        # print('EXAMPLE: ', example)
        if mode == 'language_modeling':
            text = tokenizer.bos_token + example["context"] + tokenizer.eos_token
            tokenized_text = tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors='pt')
            tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze()
            tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze()
            tokenized_text['labels'] = tokenized_text['input_ids'].clone()
            tokenized_text['labels'][tokenized_text['labels'] == tokenizer.pad_token_id] = -100
            if moe:
                routing = get_routing(text, tokenized_text)
                tokenized_text['input_routing'] = routing
            return tokenized_text

    columns = list(dataset.features) # Changed for robustness
    dataset = dataset.map(tokenization, remove_columns=columns)
    return DataLoader(
            dataset,
            collate_fn=default_data_collator,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory = True
        )

def convert_moe(model, num_experts=2):
    for i in range(len(model.transformer.h)):
        if i % 2 == 0:
            # weights of everything in the mlp layer
            weights = (model.transformer.h[i].mlp.c_fc.weight.data, 
                       model.transformer.h[i].mlp.c_fc.bias.data, 
                       model.transformer.h[i].mlp.c_proj.weight.data, 
                       model.transformer.h[i].mlp.c_proj.bias.data)
            assert len(weights) == len(list(model.transformer.h[i].mlp.parameters()))

            model.transformer.h[i].mlp = MoE(768, num_experts=num_experts, config=model.config, expert_init_weights=weights)

# trainer class
class NamedEntityRecognition:
    def __init__(self):
        self.tagger = spacy.load('en_core_web_sm')

    def __call__(self, text):
        '''
        Given a string returns a list of NER tags
        No entity is represented as an empty string
        '''
        doc = self.tagger(text)
        tagging = torch.zeros(len(text))
        # words = [token.text for token in doc]
        # tags = [token.ent_type_ for token in doc]
        for ent in doc.ents:
            tagging[ent.start_char:ent.end_char] = 1

        return tagging

class Trainer(object):
    def __init__(
        self,
        args,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        max_seq_len = 128,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        num_samples = None,
        eval_every = 1000,
        results_folder = './results',
        amp = False,
        split_batches = True,
        seed=42,
        moe=False,
    ):
        super().__init__()


        set_seeds(seed)

        self.args = args

        self.best_val_metric = 0
        self.num_samples = num_samples
        self.moe = moe

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'no',
            log_with='wandb'
        )

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.accelerator.native_amp = amp
        if not self.moe:
            self.lm = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            self.lm = GPT2LMHeadModelMoE.from_pretrained('gpt2')
            convert_moe(self.lm, num_experts=2)

        # reset seeds after model initialization
        set_seeds(seed+1)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
        num_added_tokens = self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        assert num_added_tokens == 1
        self.lm.resize_token_embeddings(len(self.tokenizer))

        self.eval_every = eval_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_num_steps = train_num_steps
        self.max_seq_len = max_seq_len
        
        # dataset and dataloader
        self.dataset = load_dataset(dataset_name)

        # self.dataset = dataset.shuffle(seed=seed)

        self.dataloader = get_dataloader(self.dataset['train'], self.tokenizer, train_batch_size, self.max_seq_len, shuffle=True, moe=self.moe)
        self.val_dataloader = get_dataloader(self.dataset['validation'].select(range(100)), self.tokenizer, eval_batch_size, self.max_seq_len, moe=self.moe)

        # optimizer

        self.opt = get_adamw_optimizer(self.lm.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.lm, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(self.lm, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.lm),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.lm)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def validation(self, num_samples=None, test=False, timing=False, seed=42):
        num_samples = default(num_samples, self.num_samples)
        self.lm.eval()
        device = self.accelerator.device
        accelerator = self.accelerator
        # Extract references
        reference_texts = {}
        if test:
            reference_texts['test'] = self.dataset['test']['text'][:num_samples]
        else:
            reference_texts['val'] = self.dataset['valid']['text'][:num_samples]
        reference_texts['train'] = self.dataset['train']['text'][:num_samples]

        torch.manual_seed(seed)
        generate_kwargs = {
            'nucleus_95':
            {'max_length':self.max_seq_len, 'min_length':5, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},}

        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in generate_kwargs.items()}  
        # Loop until enough senetences have been generated across all strategies
        input_prompt = self.tokenizer.bos_token
        tokenized_prompt = self.tokenizer(input_prompt, return_tensors="pt")
        input_ids = tokenized_prompt.input_ids.to(device)
        attention_mask = tokenized_prompt.attention_mask.to(device)
        if timing:
            start = timeit.default_timer()
        for k in  all_texts_lists.keys():
            print(f'Sampling strategy {k}')
            while len(all_texts_lists[k]) < num_samples:
                batches = num_to_groups(num_samples-len(all_texts_lists[k]), self.eval_batch_size)
                for batch in batches:
                    batch_input_ids = repeat(input_ids, '1 1 -> b 1', b=batch)
                    batch_attention_mask = repeat(attention_mask, '1 1 -> b 1', b=batch)
                    sample_ids = self.lm.generate(input_ids=batch_input_ids, attention_mask=batch_attention_mask, pad_token_id=self.tokenizer.pad_token_id, **generate_kwargs[k])
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    all_texts_lists[k].extend([text.strip() for text in texts_list if len(text.strip())>0])
                    # Exit early for beam search
                    if k == 'beam':
                        all_texts_lists[k] = [all_texts_lists[k][0]]*num_samples
                        break
        if timing:
            end = timeit.default_timer()
            wandb.run.summary[f"{num_samples}samples_time"] = end-start
            return
            
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k:v[:num_samples] for k,v in all_texts_lists.items()} 

        metrics = {}

        self.lm.to('cpu')
        torch.cuda.empty_cache() 
        for strategy, all_texts_list in text_generations.items():
            metrics[f"model/{strategy}/perplexity"] = evaluation.compute_perplexity(all_texts_list)
            metrics[f"model/{strategy}/unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            ngram_metrics = evaluation.compute_diversity(all_texts_list)
            for k, v in ngram_metrics.items():
                metrics[f"model/{strategy}/{k}"] = v
            metrics[f"model/{strategy}/memorization"] = evaluation.compute_memorization(all_texts_list, self.dataset['train']['text'])
            table = wandb.Table( 
                columns=['Samples'], data=[[text] for text in all_texts_list])
            accelerator.log({f"model/{strategy}/samples": table}, self.step)

            # Only evaluate MAUVE if generations are reasonable
            if metrics[f"model/{strategy}/perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large"]:
                for key, reference_text in reference_texts.items():
                    metrics[f"model/{strategy}/{mauve_model_id}_{key}_mauve"], _ = evaluation.compute_mauve(all_texts_list, reference_text, mauve_model_id)

        if test:
            metrics_dict = {**metrics,**self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}':v for k,v in metrics_dict.items()}
            accelerator.log(metrics_dict, self.step)
            print(metrics_dict)
        else:
            accelerator.log({**metrics}, self.step)
        torch.cuda.empty_cache() 
        self.lm.to(device)
        
        self.lm.train()

    def train(self):
        self.lm.train()
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                data = {k:v.to(device) for k,v in next(self.data_iter).items()}

                with self.accelerator.autocast():
                    loss = self.lm(**data).loss
                    loss = loss
                    total_loss += loss.item()

                    self.accelerator.backward(loss)


                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.lm.parameters())

                accelerator.clip_grad_norm_(self.lm.parameters(), 1.0)
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.lm.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            data = {k:v.to(device) for k,v in next(self.val_iter).items()}

                            with self.accelerator.autocast():
                                loss = self.lm(**data).loss
                                loss = loss
                                total_val_loss += loss.item()

                            logs = {"train/loss": total_loss, "val/loss": total_val_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step)/len(self.dataloader), "samples": self.step*self.train_batch_size}
                            pbar.set_postfix(**logs)
                             
                        self.lm.train()
                    else:
                        logs = {"train/loss": total_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step)/len(self.dataloader), "samples": self.step*self.train_batch_size}

                    accelerator.log(logs, step=self.step)


                    if self.step % self.eval_every == 0:
                        # self.validation()
                        self.save()
                        self.lm.train() 

                pbar.update(1)

        accelerator.print('training complete')

def main(args):
    trainer = Trainer(
        args=args,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        max_seq_len = args.max_seq_len,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        eval_every = args.eval_every,
        results_folder = args.output_dir,
        amp = args.amp,
        moe = args.moe,
    )

    if args.eval:
        trainer.load(args.resume_dir)
        trainer.validation(args.num_samples)
        return
    if args.eval_test:
        trainer.load(args.resume_dir)
        for seed in [42, 43, 44, 45, 46]:
            trainer.dataset = trainer.dataset.shuffle(seed)
            trainer.validation(args.num_samples, seed=seed, test=True)
        return
    if args.resume_training:
        trainer.load(args.resume_dir)

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default="squad_v2")
    parser.add_argument("--num_samples", type=int, default=250)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    # Optimization hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    # Generation Arguments
    parser.add_argument("--eval_every", type=int, default=1000)
    # Model hyperparemeters
    # Accelerate arguments
    parser.add_argument("--amp", action="store_true", default=False)
    # Load and eval model
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument("--timing", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--moe", action="store_true", default=False)
    
    args = parser.parse_args()

    if args.eval or args.resume_training or args.eval_test:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        if args.eval or args.eval_test:
            heldout_params = {'wandb_name', 'output_dir', 'resume_dir', 'eval', 'eval_test', 'ddim_sampling_eta', 'num_samples', 'sampling_timesteps'}
        else:
            heldout_params = {'output_dir', 'resume_dir', 'resume_training', 'ddim_sampling_eta', 'num_samples', 'sampling_timesteps', 'save_and_sample_every'}
        for k,v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v

    if args.output_dir is None:
        args.output_dir = get_output_dir(args)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
    main(args)