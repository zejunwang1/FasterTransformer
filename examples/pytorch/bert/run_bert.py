# Developed by Wang Zejun
import argparse
import os
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from utils.encoder import EncoderWeights, CustomEncoder
from utils.modeling_bert_v2 import BertModel, BertForQuestionAnswering, BertForSequenceClassification


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_dir', type=str, required=True)
    parser.add_argument('--text_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--ths_path', type=str, default='./lib/libth_transformer.so')
    parser.add_argument('--remove_padding', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    torch.cuda.set_device(0)

    config = AutoConfig.from_pretrained(args.hf_model_dir)
    model_name = config.architectures[0]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    model = globals()[f'{model_name}'].from_pretrained(args.hf_model_dir).cuda().half().eval()

    # Use FasterTransformer encoder
    if args.fast:
        print('Use FasterTransformer encoder...')
        pytorch_model_bin_file = os.path.join(args.hf_model_dir, 'pytorch_model.bin')
        assert os.path.isfile(pytorch_model_bin_file)
        weights = EncoderWeights(
            model.config.num_hidden_layers, model.config.hidden_size,
            torch.load(pytorch_model_bin_file, map_location='cpu')
        )
        weights.to_cuda()
        weights.to_half()
        enc = CustomEncoder(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.hidden_size // model.config.num_attention_heads,
            weights,
            remove_padding=args.remove_padding,
            path=os.path.abspath(args.ths_path)
        )
        enc_ = torch.jit.script(enc)
        model.replace_encoder(enc_)

    texts = []
    with open(args.text_file, mode='r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)

    # Warmup
    print('Warmup...')
    warmup_texts = texts[0 : args.batch_size]
    warmup_inputs = tokenizer(warmup_texts, padding=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        warmup_outputs = model(**warmup_inputs)

    n = len(texts)
    batch_size = args.batch_size
    num_batches = int((n - 1) / batch_size) + 1
    print('Running inference...')
    tic = time.time()
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_texts = texts[start : end]
        inputs = tokenizer(batch_texts, padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)

    torch.cuda.synchronize()
    toc = time.time()
    print(outputs[0])
    print('time usage: {}s'.format(toc - tic))


