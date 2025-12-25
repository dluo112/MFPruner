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
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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
    # Initialize the latency monitor.
    latency_tracker = LatencyTracker()

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    running_latencies = []
    
    data_bar = tqdm(zip(data_loader, questions), total=len(questions), 
                   desc="Processing", 
                   postfix={'avg_latency': '0.00ms', 'curr_latency': '0.00ms'})
    
    for batch_idx, ((input_ids, image_tensors, image_sizes), line) in enumerate(data_bar):
        idx = line["question_id"]
        cur_prompt = line["text"]

        question = cur_prompt
        question = question.replace("\nAnswer the question using a single word or phrase.", "")

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        # starting count time
        latency_tracker.start_timing()

        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                texts=question,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            
            if hasattr(model.model, 'visual_token_num'):
                visual_token_num = model.model.visual_token_num

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
        result_dict = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "latency_ms": current_latency if current_latency else 0,
                "visual_token_num": visual_token_num
            }
        }
        
        ans_file.write(json.dumps(result_dict) + "\n")
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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--save-latency", action="store_true", 
                       help="Whether to save detailed latency statistics")
    args = parser.parse_args()

    eval_model(args)