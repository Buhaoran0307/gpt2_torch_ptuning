from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import transformers
import argparse
import logging
import torch
import sys

from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    AutoTokenizer,
    GPT2Model,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained('gpt2/')
model = GPT2Model.from_pretrained('gpt2/')
max_seq_length = 1024 + 1024
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
logger = logging.getLogger(__name__)

transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.set_verbosity(4)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_tokens([])

def preprocess_data(examples):
        model_inputs = []

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        for i in range(len(examples)):
            if examples[i]['question'] and examples[i]['answer']:
                query, answer = examples[i]['question'], examples[i]['answer']

                prompt = "问:{}, 答:{}".format(query,answer)
                
                a_ids = tokenizer.encode(prompt)
                b_ids = tokenizer.encode(answer)

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='.\data\\train.json',
                        help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, default='.\data\\train.mindrecord',
                        help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help="The MindRecord file will be split into the number of partition. ")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length. ")
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ")
    parser.add_argument("--data_columns", type=list, default=["input_ids", "attention_mask", "labels"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. ")

    args = parser.parse_args()

    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)
    
    train_dataset = load_dataset(
        "json", 
        data_files="data/train.json",
        cache_dir='data/cache/',
    )

    train_dataset = train_dataset.map(
            preprocess_data,
            batched=True,
            desc="Running tokenizer on train dataset"
         )

    trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_result = trainer.train(device = device)
    trainer.save_state()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

# python preprocess.py --input_file data\wikitext-2\wiki.train.tokens --output_file ./wikitext-2.train..mindrecord --max_length 1025