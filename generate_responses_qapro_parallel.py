from openai import OpenAI
import argparse
import base64
from io import BytesIO
from PIL import Image
import io
import json
from datasets import load_dataset
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from evaluate_predictions import relaxed_correctness_chartqapro, fix_list_format
from collections import defaultdict
import shutil


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def get_prompt_template(strategy, category):
    """
    Returns the appropriate prompt template for a given strategy (e.g., 'direct' or 'cot') 
    and question category (e.g., 'factoid', 'multi_choice', etc.)
    """
    templates = {
        "direct": {
            "factoid": (
                "You are given a factoid question that you need to answer based on the provided image.\n\n"
                "Your answer should be a single word, number, or phrase. "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. "
                "Do not generate units. But if numerical units such as `million`, `m`, `billion`, `B`, or `K` are required, use the exact notation shown in the chart.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].\n\n"
                "Remember to generate the final answer only without any additional text!\n\n"
                "Question: {question}"
            ),
            "multi_choice": (
                "You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.\n\n"
                "Your answer should be one of the options letters only: `a`, `b`, `c` or `d` (just the letter itself without any additional text) "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].\n\n"
                "Remember to generate the final answer only without any additional text!\n\n"
                "Question: {question}"
            ),
            "hypothetical": (
                "You are given a hypothetical question that you need to answer based on the provided image.\n\n"
                "Your answer should be a single word, number, or phrase. "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. "
                "Do not generate units. But if numerical units such as `million`, `m`, `billion`, `B`, or `K` are required, use the exact notation shown in the chart.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].\n\n"
                "Remember to generate the final answer only without any additional text!\n\n"
                "Question: {question}"
            ),
            "fact_checking": (
                "You are given a fact statement that you need to assess based on the provided image.\n\n"
                "Your answer should be either `true` or `false` (without any additional text). "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].\n\n"
                "Remember to generate the final answer only without any additional text!\n\n"
                "Question: {question}"
            ),
            "conversational": (
                "You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.\n\n"
                "Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. "
                "Do not generate units. But if numerical units such as `million`, `m`, `billion`, `B`, or `K` are required, use the exact notation shown in the chart.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].\n\n"
                "Remember to generate the final answer only without any additional text!\n\n"                
                "Conversation: {conversation} Question: {question}"
            ),
        },
        "cot": {
            "factoid": (
                "You are given a factoid question that you need to answer based on the provided image.\n\n"
                "You need to think step-by-step, but your final answer should be a single word, number, or phrase. "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. "
                "Do not generate units. But if numerical units such as `million`, `m`, `billion`, `B`, or `K` are required, use the exact notation shown in the chart.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2']. ."
                "Remember to think step-by-step and format the final answer in a separate sentence like 'The answer is X'\n\n"
                "Question: {question}"
            ),
            "multi_choice": (
                "You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.\n\n"
                "You need to think step-by-step, but your final answer should be one of the options letters only: `a`, `b`, `c` or `d` (just the letter itself without any additional text). "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2']. ."
                "Remember to think step-by-step and format the final answer in a separate sentence like 'The answer is X'\n\n"
                "Question: {question}"
            ),
            "hypothetical": (
                "You are given a hypothetical question that you need to answer based on the provided image.\n\n"
                "You need to think step-by-step, but your final answer should be a single word, number, or phrase. "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. "
                "Do not generate units. But if numerical units such as `million`, `m`, `billion`, `B`, or `K` are required, use the exact notation shown in the chart.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2']. ."
                "Remember to think step-by-step and format the final answer in a separate sentence like 'The answer is X'\n\n"
                "Question: {question}"
            ),
            "fact_checking": (
                "You are given a fact statement that you need to assess based on the information in the provided image.\n\n"
                "You need to think step-by-step, but your final answer should be either true or false (without any additional text). "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2']. ."
                "Remember to think step-by-step and format the final answer in a separate sentence like 'The answer is X'\n\n"
                "Question: {question}"
            ),
            "conversational": (
                "You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.\n\n"
                "You need to think step-by-step, but your final answer should be a single word, number, or phrase. "
                "If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. "
                "Do not generate units. But if numerical units such as `million`, `m`, `billion`, `B`, or `K` are required, use the exact notation shown in the chart.\n\n"
                "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2']. ."
                "Remember to think step-by-step and format the final answer in a separate sentence like 'The answer is X'\n\n"
                "Conversation: {conversation} Question: {question}"
            ),
        }
    }
    return templates[strategy][category]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate responses for ChartQAPro dataset using VLLM model inference."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ahmed-masry/ChartQAPro",
        help="Name of the Hugging Face dataset to load"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset to use (e.g., train, test)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Cache directory for dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        choices=["Qwen/Qwen2-VL-2B", "microsoft/Phi-3.5-vision-instruct", "google/gemma-3-4b-it"],
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="cot",
        choices=["direct", "cot"],
        help="Prompt strategy to use (direct or chain-of-thought)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True, 
        help="Path to save responses"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of samples to process (None for all)"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8055/v1",
        help="Base URL for the API server"
    )

    parser.add_argument(
        "--skip_question_types",
        type=str,
        nargs='+',
        default=[],
        choices=["Factoid", "Multi Choice", "Hypothetical", "Fact Checking", "Conversational"],
        help="Question types to skip (e.g., --skip_question_types Factoid Hypothetical)"
    )
    parser.add_argument(
        "--only_question_types",
        type=str,
        nargs='+',
        default=[],
        choices=["Factoid", "Multi Choice", "Hypothetical", "Fact Checking", "Conversational"],
        help="Only process these question types (e.g., --only_question_types Conversational Hypothetical)"
    )  
    parser.add_argument(
        "--use_cepo",
        action='store_true',
        default=False
    )  
    parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="Number of parallel workers (default: 2)"
    )   
    

    return parser.parse_args()


def extract_final_answer(model_output):
    """Extract just the final answer from CoT reasoning"""
    # Look for "The answer is X" pattern
    answer_pattern = re.search(r"[Tt]he answer is\s+(.+?)$", model_output)
    if answer_pattern:
        return answer_pattern.group(1).strip()
    
    # Look for "The answer: X" pattern
    answer_pattern2 = re.search(r"[Tt]he answer:\s+(.+?)$", model_output)
    if answer_pattern2:
        return answer_pattern2.group(1).strip()
    
    # If no pattern matches, return the last sentence (could be the answer)
    # We should NOT reach here, but it's a fallback
    sentences = model_output.split('.')
    if sentences:
        return sentences[-1].strip()
    
    return model_output  # Fallback to original

def encode_base64_content_pil_image(image) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    # Handle both PIL Image objects and bytes
    if isinstance(image, bytes):
        # Convert bytes to PIL Image
        image_pil = Image.open(io.BytesIO(image))
        temp_img = Image.open(io.BytesIO(image))
        img_width, img_height = temp_img.size
        print(f"Image resolution: {img_width}x{img_height} pixels")
    else:
        image_pil = image
        temp_img = Image.open(io.BytesIO(image))
        img_width, img_height = temp_img.size
        print(f"Image resolution: {img_width}x{img_height} pixels")

    image_pil = image_pil.convert("RGB")
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{img_b64}"
    return image_url


def load_huggingface_dataset(dataset_name, split, cache_dir=None):
    """Load the ChartQAPro dataset."""
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    return ds[split]

def create_client(api_base):
    """Create OpenAI client for VLLM server."""
    openai_api_key = "serving-on-vllm"  # Placeholder key for VLLM
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
        timeout=None
    )
    return client

def map_question_type(question_type):
    """Map the question type from the dataset to the template category.
    To overcome key not found errors, we have 
    """
    mapping = {
        "factoid": "factoid",
        "Factoid": "factoid",
        "multi-choice": "multi_choice",
        "Multi-choice": "multi_choice", 
        "multiple-choice": "multi_choice", 
        "Multiple-choice": "multi_choice", 
        "hypothetical": "hypothetical",
        "Hypothetical": "hypothetical", 
        "fact-checking": "fact_checking",
        "Fact-checking": "fact_checking", 
        "conversational": "conversational",
        "Conversational": "conversational", 
    }
    
    # Normalize the question type (lowercase, remove spaces)
    normalized_type = question_type.lower().replace(" ", "-")
    
    # Default to factoid if no match is found
    return mapping.get(normalized_type, "factoid")

def process_sample(idx, sample, args, client, total_samples, cepo_dir=None):
    """Process a single sample"""
    # Create a string buffer to capture logs for this sample
    log_capture = io.StringIO()
    sample_handler = logging.StreamHandler(log_capture)
    sample_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logging.getLogger().addHandler(sample_handler)

    # Use the original dataset index for the sample_id
    original_idx = sample["original_dataset_idx"]
    sample_id = f"{original_idx:04d}"  
    logging.info(f"----- Processing sample {idx+1}/{total_samples} (ID: {sample_id}) ----- ")  
    
    # Extract data from the sample
    question = sample["Question"]
    gt_answer = sample["Answer"]
    question_type = sample["Question Type"]
    image = sample["image"]
    paragraph = sample.get("Paragraph", "")
    year_flags = sample["Year"]
    
    category = map_question_type(question_type)
    
    # Format the question appropriately
    if isinstance(question, list):
        if category == "conversational":
            question_text = question[-1]
            if len(question) > 1:
                conversation_pairs = []
                for i in range(len(question)-1):
                    conversation_pairs.append(f"Q: {question[i]}")
                    if isinstance(gt_answer, list) and i < len(gt_answer)-1:
                        conversation_pairs.append(f"A: {gt_answer[i]}")
                conversation_context = "\n".join(conversation_pairs)
                paragraph = conversation_context + "\n" + paragraph if paragraph else conversation_context
        else:
            question_text = question[0]
    else:
        question_text = question
    
    # Get prompt template and format it
    try:
        template = get_prompt_template(args.strategy, category)
        if category == "conversational":
            prompt_text = template.format(conversation=paragraph, question=question_text)
        else:
            prompt_text = template.format(question=question_text)
        
        logging.info(f"\n=== PROMPT TO MODEL ({category}, {args.strategy}) ===\n{prompt_text}\n=== END PROMPT ===\n")
    except KeyError:
        logging.warning(f"Template formatting failed for {category}, using factoid template")
        factoid_template = get_prompt_template(args.strategy, "factoid")
        prompt_text = factoid_template.format(question=question_text)
    
    # Process the image and generate response
    try:
        img_b64 = encode_base64_content_pil_image(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        start_time = time.time()
        if args.use_cepo:
            assert cepo_dir is not None, "CePO directory must be specified when using CePO"
            cepo_log_file = os.path.join(cepo_dir, f"cepo_sample_{sample_id}.json")
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                extra_body={"log_file": cepo_log_file}
            )
        else:
            cepo_log_file = None
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512
            )

        
        elapsed_time = time.time() - start_time
        model_output = response.choices[0].message.content.strip()
        
        # Log token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        logging.info(f"Actual token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
        
        logging.info(f"Success! Time: {elapsed_time:.2f}s")
        logging.info(f"Strategy: {args.strategy}")
        logging.info(f"Question Type: {question_type}")
        logging.info(f"Model output: '{model_output}'")
        
        # Process the model output - moved inside try block
        if args.strategy == "cot":
            cleaned_output = extract_final_answer(model_output)
            logging.info(f"Extracted final answer: '{cleaned_output}'")
        else:
            cleaned_output = model_output

        if isinstance(year_flags, list) and len(year_flags) > 0:
            year_flags_for_eval = year_flags
            if category == 'conversational' and isinstance(year_flags, list) and len(year_flags) > 1:
                year_flags_for_eval = year_flags[-1:]
            
            always_use_exact_match = True if category in ['fact_checking', 'multi_choice'] else False
            accuracy_score = relaxed_correctness_chartqapro(
                gt_answer, 
                cleaned_output, 
                year_flags=year_flags_for_eval,
                always_use_exact_match=always_use_exact_match
            )
            is_correct = accuracy_score > 0.0  # Consider any non-zero score as partially correct
        else:
            is_correct = False
            
        detailed_result = {
            "prompt_text": prompt_text,
            "Question": question_text,
            "Answer": gt_answer,
            "Question Type": question_type,
            "raw_model_output": model_output,
            "prediction": cleaned_output,
            "image_url": img_b64,
            "is_correct": is_correct,
            "accuracy_score": accuracy_score if 'accuracy_score' in locals() else None,
            "original_idx": original_idx
        }
        streamlined_result = {
            "Answer": gt_answer,
            "Question Type": question_type,
            "Year": year_flags,
            "prediction": cleaned_output,
            "original_idx": original_idx, # Keep for sorting
            "is_correct": is_correct,
            "accuracy_score": accuracy_score if 'accuracy_score' in locals() else None,
        }
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error: {error_msg[:150]}")
        model_output = "Error occured"
        cleaned_output = "unanswerable"
        if 'image_b64' not in locals():
            img_b64 = "image processing failed"
        detailed_result = {
            "prompt_text": prompt_text,
            "Question": question_text,
            "Answer": gt_answer,
            "Question Type": question_type,
            "raw_model_output": model_output,
            "prediction": cleaned_output,
            "image_url": img_b64,
            "original_idx": idx,
            "is_correct": False,
            "accuracy_score": 0.0
        }
        streamlined_result = {
            "Answer": gt_answer,
            "Question Type": question_type,
            "Year": year_flags,
            "prediction": cleaned_output,
            "original_idx": idx, # Keep for sorting
            "is_correct": False,
            "accuracy_score": 0.0
        }

    # Capture and save the log for this sample
    logs = log_capture.getvalue()
    logging.getLogger().removeHandler(sample_handler)
    
    return detailed_result, streamlined_result, logs, sample_id, question_type, cepo_log_file

def generate_responses(args):
    """Main function to generate responses for ChartQAPro dataset."""
    full_ds = load_huggingface_dataset(args.dataset_name, args.split, args.cache_dir)

    # Add original indices to the dataset before filtering
    ds_with_indices = full_ds.map(
        lambda example, idx: {"original_dataset_idx": idx},
        with_indices=True
    )

    ds = ds_with_indices
    # Filter by question types if specified
    if args.skip_question_types:
        logging.info(f"Skipping question types: {args.skip_question_types}")
        original_size = len(ds)
        ds = ds.filter(lambda x: x["Question Type"] not in args.skip_question_types)
        logging.info(f"Filtered dataset: {original_size} -> {len(ds)} samples")
    
    if args.only_question_types:
        logging.info(f"Only processing question types: {args.only_question_types}")
        original_size = len(ds)
        ds = ds.filter(lambda x: x["Question Type"] in args.only_question_types)
        logging.info(f"Filtered dataset: {original_size} -> {len(ds)} samples")
    
    if args.sample_limit:
        ds = ds.select(range(min(args.sample_limit, len(ds))))
    
    # Create output directory structure
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model_name_safe = args.model.replace("/", "_").lower()
    output_dir_model = os.path.join(args.output_dir, model_name_safe)
    if not os.path.exists(output_dir_model):
        os.makedirs(output_dir_model)
    
    # Create subdirectories for individual results and logs
    results_dir = os.path.join(output_dir_model, f"{args.strategy}_{args.split}_results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cepo_dir = None
    if args.use_cepo:
        logging.info("Using CEPo for response generation")
        cepo_dir = os.path.join(results_dir, "cepo")
        if not os.path.exists(cepo_dir):
            os.makedirs(cepo_dir)

    
    # # Add filter info to filename
    # filter_suffix = ""
    # if args.skip_question_types:
    #     filter_suffix = f"_skip_{'_'.join(args.skip_question_types)}"
    # elif args.only_question_types:
    #     filter_suffix = f"_only_{'_'.join(args.only_question_types)}"

    # output_file = os.path.join(results_dir, f"{args.strategy}_{args.split}{filter_suffix}.json")
    
    # Create client
    client = create_client(args.api_base)
    
    total_samples = len(ds)
    logging.info(f"Processing {total_samples} samples with model {args.model} using {args.strategy} strategy")
    
    # Set number of workers
    num_workers = args.num_workers
    logging.info(f"Using {num_workers} parallel workers")
    
    all_detailed_results = []
    all_streamlined_results = []
    
    # Process samples in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_sample, idx, sample, args, client, total_samples, cepo_dir): idx
            for idx, sample in enumerate(ds)
        }
        
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                # Correctly unpack the tuple returned by process_sample
                detailed_result, streamlined_result, logs, sample_id, question_type, cepo_log_file = future.result()
                
                type_dir = question_type.replace(" ", "_")
                type_results_dir = os.path.join(results_dir, type_dir)
                if not os.path.exists(type_results_dir):
                    os.makedirs(type_results_dir)
                

                result_file = os.path.join(type_results_dir, f"sample_{sample_id}.json")
                with open(result_file, "w") as f:
                    json.dump(detailed_result, f, indent=2)
                
                if cepo_log_file is not None:
                    # Move the CePO log file to the type-specific directory
                    shutil.move(cepo_log_file, os.path.join(type_results_dir, os.path.split(cepo_log_file)[-1]))
        
                all_streamlined_results.append(streamlined_result)
                completed += 1
                logging.info(f"✓ Completed {completed}/{total_samples} (Sample {idx+1})")
            except Exception as e:
                logging.error(f"Sample {idx+1} generated an exception: {e}")
                # Add a placeholder for failed samples
                result = {
                    "Answer": ds[idx]["Answer"],
                    "Question Type": ds[idx]["Question Type"],
                    "Year": ds[idx]["Year"],
                    "prediction": "unanswerable",
                    "original_idx": idx
                }
                all_streamlined_results.append(result)
            
    
    # Sort results by original index to maintain ordering
    all_streamlined_results.sort(key=lambda x: x.get("original_idx", float('inf')))

    # group by question type and dump separate jsons for each subset
    grouped = defaultdict(list)
    for item in all_streamlined_results:
        grouped[item["Question Type"]].append(item)

    grouped = dict(grouped)

    for question_type, items in grouped.items():
        output_file = os.path.join(results_dir, f"{args.strategy}_{args.split}_results_{question_type}.json")
        with open(output_file, "w") as f:
            json.dump(items, f, indent=2)
            logging.info(f"Saved combined results to {output_file} ({len(items)} samples)")
    
    # # Save combined results
    # with open(output_file, "w") as f:
    #     json.dump(all_streamlined_results, f, indent=2)
    # logging.info(f"Saved combined results to {output_file} ({len(all_streamlined_results)} samples)")
    logging.info(f"Individual results saved to {results_dir}")
    
if __name__ == "__main__":
    args = parse_arguments()
    generate_responses(args)