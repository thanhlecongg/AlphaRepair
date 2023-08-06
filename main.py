import os
import re
import subprocess
import torch
import argparse
import csv
import ast
from io import StringIO
import time
from transformers import RobertaTokenizer, RobertaForMaskedLM
import datetime

from simple_template import generate_template, remove_redudant, generate_match_template, match_simple_operator
from tool.logger import Logger
from tool.fault_localization import get_location
from tool.d4j import build_d4j1_2
from validate_patches import GVpatches, UNIAPRpatches
from bert_beam_search import BeamSearch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def add_new_line(file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret_before = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    post_code = data[line_loc:]
    old_code = data[line_loc].strip()
    masked_line = " " + mask_token * 20 + " "
    line_size = 100
    while (1):
        pre_code_input = "</s> " + " ".join(
            [x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + masked_line + post_code_input, return_tensors='pt')['input_ids'].size()[1] < 490:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))  # actual context len =  2*line_size
    print("Context Before:\n{}".format(pre_code_input))
    print("Context After:\n{}".format(post_code_input))

    # Straight up line replacement
    for token_len in range(1, 30):  # Within 10

        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret_before.append(("".join(beam[2]), beam[0] / token_len, "Before " + masked_line))
    ret_before.sort(key=lambda x: x[1], reverse=True)
    ret_before = remove_redudant(ret_before)

    ret = []
    ret.extend(ret_before)
    ret.sort(key=lambda x: x[1], reverse=True)

    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code


def process_file(file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()
    ret = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    fault_line = comment_remover(data[line_loc].strip())  # remove comments
    old_code = data[line_loc].strip()
    post_code = data[line_loc + 1:]

    line_size = 100

    while (1):
        pre_code_input = "</s> " + " ".join([x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + fault_line + post_code_input, return_tensors='pt')['input_ids'].size()[1] < 490:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))  # actual context len =  2*line_size
    print("Context Before:\n{}".format(pre_code_input))
    print(">> {} <<".format(fault_line))
    print("Context After:\n{}".format(post_code_input))

    fault_line_token_size = tokenizer(fault_line, return_tensors='pt')["input_ids"].shape[1] - 2

    # Straight up line replacement
    for token_len in range(fault_line_token_size - 5, fault_line_token_size + 5):  # Within 10
        if token_len <= 0:
            continue
        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret.append(("".join(beam[2]), beam[0] / token_len, masked_line))

    templates = generate_template(fault_line)
    match_template = generate_match_template(fault_line, tokenizer)
    simple_operator_template = match_simple_operator(fault_line, tokenizer)
    n_template = 0
    for partial_beginning, partial_end in templates:
        for token_len in range(2, 11):
            if token_len <= 0:
                continue
            n_template += 1
    for match, length in match_template:
        for token_len in range(1, length + 5):
            n_template += 1
    for template in simple_operator_template:
            n_template += 1
                
    for partial_beginning, partial_end in templates:
        temp_size = fault_line_token_size - (
                tokenizer(partial_beginning, return_tensors='pt')["input_ids"].shape[1] - 2) - (
                            tokenizer(partial_end, return_tensors='pt')["input_ids"].shape[1] - 2)
        for token_len in range(2, 11):
            if token_len <= 0:
                continue
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len, masked_line))

    for match, length in match_template:
        for token_len in range(1, length + 5):
            if len(match.split(mask_token)) == 2:
                masked_line = " " + match.split(mask_token)[0] + mask_token * token_len + match.split(mask_token)[
                    1] + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    ret.append((match.split(mask_token)[0] + "".join(beam[2]) + match.split(mask_token)[1],
                                beam[0] / token_len, masked_line))
            else:
                masked_line = " "
                masked_line += (mask_token * token_len).join(match.split(mask_token)) + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    index = 0
                    gen_line = ""
                    for c in masked_line.split(mask_token)[:-1]:
                        gen_line += c
                        gen_line += beam[2][index]
                        index += 1
                    gen_line += masked_line.split(mask_token)[-1]
                    gen_line = gen_line[1:-1]
                    ret.append((gen_line, beam[0] / (token_len * (len(match.split(mask_token)) - 1)), masked_line))

    for template in simple_operator_template:
        token_len = template.count("<mask>")
        masked_line = " " + template + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            index = 0
            gen_line = ""
            for c in masked_line.split(mask_token)[:-1]:
                gen_line += c
                gen_line += beam[2][index]
                index += 1
            gen_line += masked_line.split(mask_token)[-1]
            gen_line = gen_line[1:-1]
            ret.append((gen_line, beam[0] / token_len, masked_line))
    ret.sort(key=lambda x: x[1], reverse=True)
    ret = remove_redudant(ret)
    print("Generated Patches: " + str(len(ret)))
    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code

def save_array_to_csv(array, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(array)

def load_array_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        array = next(reader)
    return array

def convert(input_string):
    input_string = input_string[1: -1]
    print(input_string)
    # Prepare a CSV-like input for csv.reader
    csv_like_input = StringIO(input_string.replace(', ', ','))

    # Read the CSV-like input and convert to a list
    parsed_list = next(parsed_list = next(csv.reader([input_string], quotechar="'", delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)))

    # Process each element to convert to appropriate types
    converted_list = []
    for item in parsed_list:
        try:
            converted_item = ast.literal_eval(item)
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, keep it as a string
            converted_item = item
        converted_list.append(converted_item)
    return converted_list

def current_formatted_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%a %d %b %Y %H:%M:%S %p")

   
def repair(source_dir, buggy_file, buggy_loc, beam_width, re_rank, top_n_patches, out_dir):
    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    subprocess.run(f'rm -rf {out_dir}/*', shell=True)
    patch_pool_folder = out_dir
    os.makedirs(patch_pool_folder, exist_ok=True)
    subprocess.run(f'rm -rf {patch_pool_folder}/*', shell=True)
    
    start_time = time.time()

    pre_code, fault_line, changes, post_code = process_file(os.path.join(source_dir, buggy_file), buggy_loc, tokenizer, model, beam_width,
                                                                        re_rank, top_n_patches)
    
    end_time = time.time()
    print(f"Patch Generation Time: {end_time - start_time}s")
    save_array_to_csv(pre_code, os.path.join(patch_pool_folder, "pre_code.csv"))
    save_array_to_csv(changes, os.path.join(patch_pool_folder, "changes.csv"))
    save_array_to_csv(post_code, os.path.join(patch_pool_folder, "post_code.csv"))
    open(os.path.join(patch_pool_folder, "fault_line.txt"), "w").write(fault_line)

def validate(bug_id, buggy_file, buggy_loc, uniapr, source_dir, out_dir):
    logger = Logger(os.path.join(out_dir, bug_id + "_result.txt"))
    testmethods = os.popen('defects4j export -w %s -p tests.trigger' % source_dir).readlines()
    patch_pool_folder = out_dir
    pre_code = load_array_from_csv(os.path.join(patch_pool_folder, "pre_code.csv"))
    changes = load_array_from_csv(os.path.join(patch_pool_folder, "changes.csv"))
    changes = [convert(c) for c in changes]
    post_code = load_array_from_csv(os.path.join(patch_pool_folder, "post_code.csv"))
    fault_line = open(os.path.join(patch_pool_folder, "fault_line.txt"), "r").read()
    if uniapr:
        raise NotImplementedError
    else:
        validator = GVpatches(bug_id, testmethods, logger, source_dir, patch_pool_folder=patch_pool_folder)
    
    validator.add_new_patch_generation(pre_code, fault_line, changes, post_code, buggy_file, buggy_loc, 0)
    validator.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--bug_id', type=str, default='Chart-1')
    parser.add_argument('--src_dir', type=str, default='test_prj')
    parser.add_argument('--buggy_file', type=str, default='source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java')
    parser.add_argument('--buggy_loc', type=int, default=1797)
    parser.add_argument('--uniapr', action='store_true', default=False)
    parser.add_argument('--output_folder', type=str, default='codebert_result')
    parser.add_argument('--skip_v', action='store_true', default=False)
    parser.add_argument('--re_rank', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--top_n_patches', type=int, default=-1)
    print("Start Time: " + current_formatted_time())
    args = parser.parse_args()
    print("Run with setting:")
    print(args)
    print(f"Using device: {device}")
    if args.task == "repair":
        repair(args.src_dir, args.buggy_file, args.buggy_loc - 1, args.beam_width, args.re_rank, args.top_n_patches, args.output_folder)
    elif args.task == "validate":
        validate(args.bug_id, args.buggy_file, args.buggy_loc - 1, args.uniapr, args.src_dir, args.output_folder)
    print("End Time: " + current_formatted_time())
