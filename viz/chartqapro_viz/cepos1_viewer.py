import streamlit as st
import pandas as pd
import os
import numpy as np

# Function to get folder path
def get_folder_path():
    folder_path = st.text_input("Enter the path to the folder:", placeholder=f"Type something")
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

    problem_counts_df['value_counts_list_cumsum'] = problem_counts_df.apply(lambda row: np.cumsum(row["value_counts_list"]), axis=1)
    
    return problem_counts_df, max_reward


def add_sample_plan_filter(problem_counts_df, max_reward):

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
    

    def choose_fcn(row):
        value_counts_list_cumsum = var1 = row.value_counts_list_cumsum
        op_dict = {"atleast": ">=", "atmost": "<="}
        op1 = op_dict[st.session_state.get('atleast_atmost_cond1')]
        op2 = op_dict[st.session_state.get('atleast_atmost_cond2')]

        # process condition 1:
        bool_cond1_str  = f"var1[{st.session_state.get('num_rewards_1')}] {op1} {st.session_state.get('num_plans_cond1')}"
        bool_cond1 = eval(bool_cond1_str)

        # process condition 2:
        bool_cond2_str  = f"var1[{st.session_state.get('num_rewards_2')}] {op2} {st.session_state.get('num_plans_cond2')}"
        bool_cond2 = eval(bool_cond2_str)

        final_bool = f"{bool_cond1}, {bool_cond1_str} {st.session_state.get('and_or_op')} {bool_cond2_str} {bool_cond2}"

        return final_bool


    problem_counts_df["choose"] = problem_counts_df.apply(lambda row: choose_fcn(row), axis=1)

    st.dataframe(problem_counts_df) 




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

    
    add_sample_plan_filter(problem_counts_df, max_reward)



if __name__ == "__main__":
    main()    




