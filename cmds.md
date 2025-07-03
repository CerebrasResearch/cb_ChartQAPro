
```bash
vllm serve google/gemma-3-4b-it --tensor-parallel-size=8 --max-num-seqs 1 --enable-chunked-prefill --gpu-memory-utilization 0.98 --enforce-eager --limit_mm_per_prompt "image=1"
```



```bash
python /mnt/local/shared/aarti/cb_optillm_internal/optillm/chartqapro/generate_responses_qapro_parallel.py --model "gemma-3-4b-it" --strategy "direct" --api_base http://127.0.0.1:8010/v1 --output_dir /mnt/local/shared/aarti/cb_optillm_internal/optillm/chartqapro/results_gemma4b_direct_cepo_bon1 --only_question_types Factoid
```





```bash
curl -X POST "http://127.0.0.1:8010/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer serving-on-vllm"   -d '{
    "model": "gemma-3-4b-it",
    "messages": [
      
          {
              "role": "system",
              "content": "You are a helpful assistant."
          },
          {
              "role": "user",
              "content": "What is the capital of Canada?"
          }
      ], 
    "max_tokens": 100,
    "temperature": 0.7
  }'
```



```bash
rsync -rzvhP 


rsync -rzvhP -e "ssh -i /cb/home/aarti/.ssh/mlftmp_private_key" /cb/home/aarti/ws/code/ws_repos/cb_optillm_internal/optillm/cepo/chartqapro/evaluate_predictions.py aarti@mlftmp1:/mnt/local/shared/aarti/cb_optillm_internal/optillm/chartqapro


```







```bash
vllm serve google/gemma-3-4b-it \
--tensor-parallel-size 4 \
--api-key serving-on-vllm \
--port 8055 \
--host 127.0.0.1 \
--max-num-seqs 1 \
--enable-chunked-prefill \
--gpu-memory-utilization 0.98 \
--enforce-eager \
--limit_mm_per_prompt "image=1"
```





```bash
curl -X POST "http://127.0.0.1:8055/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer serving-on-vllm"   -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [
      
          {
              "role": "system",
              "content": "You are a helpful assistant."
          },
          {
              "role": "user",
              "content": "What is the capital of Canada?"
          }
      ], 
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

```bash
/cb/home/utkuu/ws/miniconda3/envs/vllm/bin/python /cb/home/aarti/ws/code/ws_repos/chartqapro/generate_responses_qapro_parallel.py --model "google/gemma-3-4b-it" --strategy "direct" --api_base http://127.0.0.1:8055/v1 --output_dir /cb/home/aarti/ws/code/ws_repos/chartqapro/results/results_gemma4b_direct --only_question_types Hypothetical --sample_limit 20
```



```bash
/cb/home/utkuu/ws/miniconda3/envs/vllm/bin/python /cb/home/aarti/ws/code/ws_repos/chartqapro/generate_responses_qapro_parallel.py --model "google/gemma-3-4b-it" --strategy "cot" --api_base http://127.0.0.1:8055/v1 --output_dir /cb/home/aarti/ws/code/ws_repos/chartqapro/results/results_gemma4b_cot --only_question_types Hypothetical --sample_limit 20
```



```bash
/cb/home/utkuu/ws/miniconda3/envs/vllm/bin/python /cb/home/aarti/ws/code/ws_repos/chartqapro/generate_responses_qapro_parallel.py --model "google/gemma-3-4b-it" --strategy "direct" --api_base http://127.0.0.1:8055/v1 --output_dir /cb/home/aarti/ws/code/ws_repos/chartqapro/results/results_gemma4b_direct --only_question_types Factoid --sample_limit 20
```



```bash
/cb/home/utkuu/ws/miniconda3/envs/vllm/bin/python /cb/home/aarti/ws/code/ws_repos/chartqapro/generate_responses_qapro_parallel.py --model "google/gemma-3-4b-it" --strategy "cot" --api_base http://127.0.0.1:8055/v1 --output_dir /cb/home/aarti/ws/code/ws_repos/chartqapro/results/results_gemma4b_cot --only_question_types Factoid --sample_limit 20
```





```bash
/cb/home/utkuu/ws/miniconda3/envs/vllm/bin/python /cb/home/aarti/ws/code/ws_repos/chartqapro/generate_responses_qapro_parallel.py --model "google/gemma-3-4b-it" --strategy "direct" --api_base http://127.0.0.1:8055/v1 --output_dir /cb/home/aarti/ws/code/ws_repos/chartqapro/results/results_gemma4b_direct_allsubsets --sample_limit 10
```



```bash 
source /cb/home/aarti/envs/optillm_venv/bin/activate
```



```bash
OPENAI_API_KEY=serving-on-vllm python3 optillm.py \
  --base-url http://127.0.0.1:8055/v1 \
  --approach cepo \
  --port 8010 \
  --cepo_config_file /cb/home/aarti/ws/code/ws_repos/cb_optillm_internal/optillm/cepo/configs/cepo_config.yaml
  
```



```bash
curl -X POST "http://127.0.0.1:8010/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer serving-on-vllm"   -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [
      
          {
              "role": "system",
              "content": "You are a helpful assistant."
          },
          {
              "role": "user",
              "content": "What is the capital of Canada?"
          }
      ], 
    "max_tokens": 100,
    "temperature": 0.7
  }'
```



```bash

```



```bash
/cb/home/utkuu/ws/miniconda3/envs/vllm/bin/python /cb/home/aarti/ws/code/ws_repos/chartqapro/generate_responses_qapro_parallel.py --model "google/gemma-3-4b-it" --strategy "direct" --api_base http://127.0.0.1:8010/v1 --output_dir /cb/home/aarti/ws/code/ws_repos/chartqapro/results/results_gemma4b_direct_cepo --only_question_types Hypothetical --sample_limit 20 --use_cepo
```





```bash
source /cb/home/aarti/envs/optillm_venv/bin/activate
python -m streamlit run /cb/home/aarti/ws/code/ws_repos/chartqapro/viz/chartqapro_viz/log_viz_app.py
```

