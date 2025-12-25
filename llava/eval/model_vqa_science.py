import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class LatencyTracker:
    def __init__(self):
        self.latencies = []
        self.start_time = None
        
    def start_timing(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        
    def end_timing(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.start_time is not None:
            latency = (time.time() - self.start_time) * 1000  
            self.latencies.append(latency)
            self.start_time = None
            return latency
        return None
    
    def get_stats(self):
        if not self.latencies:
            return {}
        return {
            'count': len(self.latencies),
            'average_ms': np.mean(self.latencies),
            'min_ms': np.min(self.latencies),
            'max_ms': np.max(self.latencies),
            'std_ms': np.std(self.latencies),
            'median_ms': np.median(self.latencies),
            'total_ms': np.sum(self.latencies)
        }
    
    def reset(self):
        self.latencies = []
        self.start_time = None

def print_latency_stats(stats, model_name):
    print("\n" + "="*60)
    print(f"Model: {model_name}")
    print("="*60)
    print(f"Total sample size: {stats.get('count', 0)}")
    print(f"Average Latency: {stats.get('average_ms', 0):.2f} ms")
    print(f"Median Latency: {stats.get('median_ms', 0):.2f} ms")
    print(f"Min Latency: {stats.get('min_ms', 0):.2f} ms")
    print(f"Max Latency: {stats.get('max_ms', 0):.2f} ms")
    print(f"Standard deviation: {stats.get('std_ms', 0):.2f} ms")
    print(f"Total inference time: {stats.get('total_ms', 0)/1000:.2f} s")
    print(f"Average throughput: {stats.get('count', 0)/(stats.get('total_ms', 1)/1000):.2f} samples/s")
    print("="*60)


def save_latency_results(stats, latencies, output_file):
    results = {
        'statistics': stats,
        'individual_latencies': latencies,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLatency result has been saved to: {output_file}")



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
    )

    latency_tracker = LatencyTracker()

    # Data
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    running_latencies = []
    
    data_bar = tqdm(questions, total=len(questions), 
                   desc="Processing", 
                   postfix={'avg_latency': '0.00ms', 'curr_latency': '0.00ms'})
    
    for i, line in enumerate(data_bar):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None
            continue
        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        question = cur_prompt.replace("<image>\n", "")
        # question = question.split('\nA. ')[0]
        # question = question.split('\n')[-1]
        question = question.replace("\nAnswer with the option's letter from the given choices directly.", "")

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        latency_tracker.start_timing()
        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                texts=question,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )
            if hasattr(model.model, 'visual_token_num'):
                visual_token_num = model.model.visual_token_num
            data_bar.set_postfix(vtn=f"{visual_token_num}")

        current_latency = latency_tracker.end_timing()
        
        if current_latency is not None:
            running_latencies.append(current_latency)

        if running_latencies:
            avg_latency = np.mean(running_latencies)
            data_bar.set_postfix({
                'avg_lat': f'{avg_latency:.1f}ms',
                'curr_lat': f'{current_latency:.1f}ms' if current_latency else 'N/A',
                'vtn': f'{visual_token_num}'
            })

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    final_stats = latency_tracker.get_stats()
    individual_latencies = latency_tracker.latencies

    if final_stats:
        print_latency_stats(final_stats, model_name)

        if args.save_latency:
            latency_output_file = args.answers_file.replace('.jsonl', '_latency.json')
            save_latency_results(final_stats, individual_latencies, latency_output_file)
    else:
        print("Failed to collect latency data.")

    print(f"\nReasoning completed! Results saved in: {args.answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--save-latency", action="store_true", 
                       help="Whether to save detailed latency statistics")
    args = parser.parse_args()

    eval_model(args)
