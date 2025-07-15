import streamlit as st
import pandas as pd
import os
import numpy as np
import math
import base64
import ast
import re

# Function to get folder path
def get_folder_path():
    folder_path = st.text_input("Enter the path to the folder:", placeholder=f"Type something", key="folder_path_input")
    if os.path.isdir(folder_path):
        st.success(f"Valid folder path: {folder_path}")
    else:
        st.warning("Invalid folder path. Please try again.")
    return folder_path


def build_combined_df(folder_path):
    

    # Specify the folder path
    folder_path = "/cb/cold2/datapod/VQA_DATASETS/scienceqa/buckets/"

    # Initialize an empty dictionary to hold the value counts
    value_counts_dict = {}
    max_reward = float('-inf')

    reward_files = []
    for filename in os.listdir(folder_path):
        if filename.startswith("reward_") and filename.endswith(".jsonl"):
            reward_files.append(filename)
    
    reward_files = sorted(reward_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Loop through the files in the specified folder
    for filename in reward_files:
        # Read the JSON file
        b = pd.read_json(os.path.join(folder_path, filename), lines=True)
        # Get the value counts of "problem"
        problem_counts = b["problem"].value_counts()
        # Extract the number from the filename
        number = filename.split("_")[1].split(".")[0]
        max_reward = max(max_reward, int(number))
        # Store the counts in the dictionary
        value_counts_dict[int(number)] = problem_counts
        value_counts_dict[f"{number}_plans"] = b.groupby("problem")["plan_id"].agg(list)

    # Create a new dataframe from the dictionary
    problem_counts_df = pd.DataFrame(dict([(k, v) for k, v in value_counts_dict.items()])).fillna(0)
    _vkeys = list(value_counts_dict.keys())
    _vkeys = [_vkey for _vkey in _vkeys if not str(_vkey).endswith('_plans')] + [_vkey for _vkey in _vkeys if str(_vkey).endswith('_plans')]
    problem_counts_df = problem_counts_df[_vkeys]

    for col in problem_counts_df.columns:
        if str(col).endswith('plans'):
            problem_counts_df[col] = problem_counts_df[col].replace(0.0, None)

    def create_value_list(row, max_reward):
        value_list = []
        for i in range(0, max_reward + 1):
            value_list.append(row.get(i, 0))  # Use get to avoid KeyError if key doesn't exist
        return value_list

    del b
    problem_counts_df['value_counts_list'] = problem_counts_df.apply(lambda row: create_value_list(row, max_reward), axis=1)

    # problem_counts_df['value_counts_list_cumsum'] = problem_counts_df.apply(lambda row: np.cumsum(row["value_counts_list"]), axis=1)
    
    return problem_counts_df, max_reward


def _add_sample_plan_filter_buttons(max_reward):
    st.header("Sample Plan Filter")
    cols = st.columns(10)
    with cols[0]:
        st.markdown("Select samples with")

    with cols[1]:
        atleast_atmost_cond1 = st.selectbox(
            "atleast/atmost",
            ["atleast", "atmost"],
            index=1,
            key="atleast_atmost_cond1"
        )
    
    with cols[2]:
        num_plans_cond1 = st.selectbox(
            "num_plans",
            list(range(1, 10)),
            index=2,
            key="num_plans_cond1"
        )
    with cols[3]:
        st.write("<=")

    with cols[4]:
        num_rewards_1 = st.selectbox(
            "rewards",
            list(range(1, max_reward + 1)),
            index=0,
            key="num_rewards_1"
        )
    with cols[5]:
        and_or_op = st.selectbox(
            "and/or",
            ["and", "or"],
            index=0,
            key="and_or_op"
        )
    
    with cols[6]:
        atleast_atmost_cond2 = st.selectbox(
            "atleast/atmost",
            ["atleast", "atmost"],
            index=0,
            key="atleast_atmost_cond2"
        )

    with cols[7]:
        num_plans_cond2 = st.selectbox(
            "num_plans",
            list(range(1, 10)),
            index=0,
            key="num_plans_cond2"
        )
    with cols[8]:
        st.write("\>=")
        
    with cols[9]:
        num_rewards_2 = st.selectbox(
            "rewards",
            list(range(1, max_reward + 1)),
            index=4,
            key="num_rewards_2"
        )
    



def add_sample_plan_filter(problem_counts_df, max_reward):

    def choose_fcn(row, max_reward=max_reward):
        value_counts_list = var1 = row.value_counts_list
        op_dict = {"atleast": ">=", "atmost": "<="}
        op1 = op_dict[st.session_state.get('atleast_atmost_cond1')]
        op2 = op_dict[st.session_state.get('atleast_atmost_cond2')]

        # process condition 1:
        bool_cond1_str  = f"sum(var1[0:{st.session_state.get('num_rewards_1')}]) {op1} {st.session_state.get('num_plans_cond1')}"
        bool_cond1 = eval(bool_cond1_str)

        # process condition 2:
        bool_cond2_str  = f"sum(var1[{st.session_state.get('num_rewards_2')}:{max_reward+1}]) {op2} {st.session_state.get('num_plans_cond2')}"
        bool_cond2 = eval(bool_cond2_str)

        final_bool_str = f"{bool_cond1}, {bool_cond1_str} {st.session_state.get('and_or_op')} {bool_cond2_str} {bool_cond2}"
        final_bool = eval(f"{bool_cond1} {st.session_state.get('and_or_op')} {bool_cond2}")

        return final_bool_str, final_bool


    problem_counts_df[["choose_str", "choose"]] = problem_counts_df.apply(lambda row: choose_fcn(row, max_reward), axis=1, result_type='expand')

    return problem_counts_df


def _plot_per_plan(plan_df, plan_id):
    if plan_df.empty:
        st.warning(f"No data found for plan_id: {plan_id}.")
    else:
        with st.expander(f"**Plan: {plan_id}**", expanded=False):
            plan_str = plan_df.iloc[0]['plan']
            st.info(f"**Plan:** {plan_str}")
            with st.expander(f"**Plan Executions**"):
                for index, row in plan_df.iterrows():
                    answer = row['answer']
                    parsed_ans = row['parsed_answer_from_execution']
                    is_correct = row['is_correct']
                    execution_id = row['execution_id']
                    execution_info = row['execution']

                    expander_label = f"Execution ID: {execution_id}"
                    if is_correct:
                        expander_label = f" ✅ Result Correct"
                    else:
                        expander_label = f" ❌ Result Incorrect"

                    
                    expander_label += f" | Ground Truth: {answer}"
                    expander_label += f" | Parsed Answer: {parsed_ans}"

                    with st.expander(f"**{expander_label}**", expanded=False):
                        st.write(f"**Execution ID:** {row['execution_id']}")
                        st.write(f"**Ground Truth Answer:** {answer}")
                        st.write(f"**Parsed Answer from Execution:** {parsed_ans}")
                        if is_correct:
                            st.success("✅ **Result: Correct**")
                        elif is_correct is False:
                            st.error("❌ **Result: Incorrect**")

                        st.info(f"**Execution Info:** {execution_info}")


def _plot_per_sample(sample):
    problem = sample.name
    problem_number = int(re.search(r"\d+", problem).group())
    plans = [x for x in sample.index if str(x).endswith("_plans")]


    csv_fileloc = os.path.join(st.session_state.folder_path_input, "analysed_plans", f"analysed_data_{problem_number}.csv")
    if os.path.exists(csv_fileloc):
        df = pd.read_csv(csv_fileloc)
        with st.expander(f"{problem}"):
            cols = st.columns(3)
            question = ast.literal_eval(df.iloc[0]['question'])
            question_str = question[1]['text']
            with cols[1]:
                image = st.image(question[0]["image_url"]["url"])
            st.info(f"**Question:** {question_str}")

            for plan in plans: # "1_plans", "2_plans" which contains list of plan ids 
                plan_ids = sample.get(plan)
                num_plans = len(plan_ids) if plan_ids is not None else 0
                with st.expander(f"**Reward_{plan}: Num Plans: {num_plans}**", expanded=False):
                    if plan_ids is None:
                        st.warning(f"No plans found for reward {plan}.")
                    else:
                        for plan_id in plan_ids:
                            plan_df = df[df['plan_id'] == plan_id]
                            _plot_per_plan(plan_df, plan_id)



def plot_individual_samples(problem_counts_df, max_reward):
    filtered_samples = problem_counts_df[problem_counts_df['choose']]
    if not filtered_samples.empty:
        st.markdown('##### Sample')
        num_samples_per_page = st.number_input(f"Num samples per page", value=5, min_value=1, max_value=20)
    
        total_pages = math.ceil(len(filtered_samples) / num_samples_per_page)
        page_numbers = list(range(1, total_pages + 1))
        page = st.selectbox("Page", page_numbers, index=0, key="page_selector")
        
        start_idx = (page - 1) * num_samples_per_page
        end_idx = min(start_idx + num_samples_per_page, len(filtered_samples))

        # Display samples for current page
        for i in range(start_idx, end_idx):
            sample = filtered_samples.iloc[i]
            
            _plot_per_sample(sample)

        # Navigation buttons
        cols_button = st.columns(3)
        if page > 1:
            cols_button[0].button("Previous Page", on_click=lambda: st.session_state.update({"page_selector": page-1}))
        cols_button[1].markdown("[Back to Top](#log-visualizer)")
        if page < total_pages:
            cols_button[2].button("Next Page", on_click=lambda: st.session_state.update({"page_selector": page+1}))
    else:  
        st.warning("No samples match the selected filter criteria.")




# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title=f"CePO S1 data visualizer", layout="wide")
    st.title("CePO S1 Data Visualizer")

    folder_path = get_folder_path()
    if folder_path:
        problem_counts_df, max_reward = build_combined_df(folder_path)
        if not problem_counts_df.empty:
            st.dataframe(problem_counts_df)
        else:
            st.warning("No data found in the specified folder.")

        _add_sample_plan_filter_buttons(max_reward)
        problem_counts_df = add_sample_plan_filter(problem_counts_df, max_reward)
        st.dataframe(problem_counts_df[problem_counts_df['choose']])
        st.info(f"Samples {len(problem_counts_df[problem_counts_df['choose']])}/{len(problem_counts_df)} selected based on the filter criteria.")

        st.download_button(
            label="Download Dataframe",
            data=problem_counts_df[problem_counts_df['choose']].to_csv().encode('utf-8'),
            file_name="chosen_data.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        plot_individual_samples(problem_counts_df, max_reward)






if __name__ == "__main__":
    main()    




