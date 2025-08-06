import os
import json
import argparse
import numpy as np
from collections import defaultdict
from evaluate_predictions import evaluate_predictions_chartqapro

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze results from multiple trials")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing trial results")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy used")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_name_safe = args.model.replace("/", "_").lower()
    
    # Question types
    question_types = ["Factoid", "Multi Choice", "Hypothetical", "Fact Checking", "Conversational"]
    
    # Dictionary to store scores across trials
    scores_by_type = defaultdict(list)
    
    # Iterate through each trial
    for i in range(1, args.trials + 1):
        trial_dir = os.path.join(args.base_dir, f"trial_{i}")
        results_dir = os.path.join(trial_dir, model_name_safe, f"{args.strategy}_test_results")
        
        # Check if results directory exists
        if not os.path.exists(results_dir):
            print(f"Warning: Results directory not found for trial {i}: {results_dir}")
            continue
        
        # Process each question type
        for question_type in question_types:
            # question_type_filename = question_type.replace(" ", "_")
            question_type_filename = question_type
            result_file = os.path.join(results_dir, f"{args.strategy}_test_results_{question_type_filename}.json")
            print(f"results_file: {result_file}")
            
            if os.path.exists(result_file):
                print(f"Processing {question_type} results for trial {i}")
                with open(result_file, 'r') as f:
                    predictions = json.load(f)
                
                # Evaluate predictions for this type
                scores = evaluate_predictions_chartqapro(predictions)
                
                # Store score for this question type
                for k, v in scores.items():
                    scores_by_type[k].append(v)
    
    # Calculate mean and standard deviation
    stats = {}
    for qtype, scores in scores_by_type.items():
        mean = np.mean(scores)
        std = np.std(scores)
        stats[qtype] = {
            "mean": mean,
            "std": std,
            "scores": scores
        }
    
    # Print results
    print("\n=== Results Summary ===")
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Number of trials: {args.trials}")
    print("\nPerformance by Question Type:")
    
    for qtype, stat in sorted(stats.items()):
        print(f"\n{qtype}:")
        print(f"  Mean accuracy: {stat['mean']*100:.2f}%")
        print(f"  Std deviation: {stat['std']*100:.2f}%")
        print(f"  Raw scores: {[f'{s*100:.2f}%' for s in stat['scores']]}")
    
    # Save results to file
    output_file = os.path.join(args.base_dir, f"{model_name_safe}_{args.strategy}_stats.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed statistics saved to {output_file}")

if __name__ == "__main__":
    main()