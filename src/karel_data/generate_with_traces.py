#!/usr/bin/env python
import os
import argparse
import numpy as np

from karel import KarelWithCurlyParser, KarelForSynthesisParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError

try:
    from tqdm import trange
except:
    trange = range

if __name__ == '__main__':
    data_arg = argparse.ArgumentParser()
    data_arg.add_argument('--num_train', type=int, default=1000000)
    data_arg.add_argument('--num_test', type=int, default=5000)
    data_arg.add_argument('--num_val', type=int, default=5000)
    data_arg.add_argument('--num_examples', type=int, default=2)
    data_arg.add_argument('--parser_type', type=str, default='curly', choices=['curly', 'synthesis'])
    data_arg.add_argument('--data_dir', type=str, default='data')
    data_arg.add_argument('--max_depth', type=int, default=5)
    data_arg.add_argument('--mode', type=str, default='token', choices=['text', 'token'])
    data_arg.add_argument('--beautify', type=str2bool, default=False)
    data_arg.add_argument('--world_height', type=int, default=8)
    data_arg.add_argument('--world_width', type=int, default=8)
    config = data_arg.parse_args()

    makedirs(config.data_dir)
    datasets = ['train', 'test', 'val']

    parser = KarelWithCurlyParser() if config.parser_type == "curly" else KarelForSynthesisParser()

    if config.mode == 'text':
        for name in datasets:
            data_num = getattr(config, f"num_{name}")
            text = ""
            text_path = os.path.join(config.data_dir, f"{name}.txt")
            for _ in trange(data_num):
                code = parser.random_code(stmt_max_depth=config.max_depth)
                if config.beautify:
                    code = beautify(code)
                text += code + "\n"
            with open(text_path, 'w') as f:
                f.write(text)
    else:
        for name in datasets:
            data_num = getattr(config, f"num_{name}")

            inputs, outputs, codes, code_lengths = [], [], [], []
            traces = []

            for _ in trange(data_num):
                while True:
                    parser.new_game(world_size=(config.world_width, config.world_height))
                    input_state = parser.get_state()

                    code = parser.random_code(stmt_max_depth=config.max_depth)

                    try:
                        # run and capture trace
                        _, trace_states = parser.run_with_trace(code, include_initial=True)
                        output_state = parser.get_state()
                    except TimeoutError:
                        continue
                    except IndexError:
                        continue

                    inputs.append(input_state)
                    outputs.append(output_state)

                    token_idxes = parser.lex_to_idx(code, details=True)
                    codes.append(token_idxes)
                    code_lengths.append(len(token_idxes))

                    # Save per-example list of state tensors (ragged)
                    traces.append(trace_states)
                    break

            # store ragged traces as object array
            npz_path = os.path.join(config.data_dir, name)
            np.savez(npz_path,
                     inputs=np.array(inputs),
                     outputs=np.array(outputs),
                     codes=np.array(codes, dtype=object),
                     code_lengths=np.array(code_lengths),
                     traces=np.array(traces, dtype=object))
