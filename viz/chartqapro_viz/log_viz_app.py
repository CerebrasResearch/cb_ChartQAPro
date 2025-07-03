import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yaml
import html
import math
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))


_RUN_JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), "log_comparison.json")

def read_run_json_file_path():
    with open(_RUN_JSON_FILE_PATH, "r") as fh:
        json_data = json.load(fh)
    return json_data


def check_path(key_session_state, cont):
    if not os.path.exists(st.session_state[key_session_state]):
        with cont:
            st.error(f"Path does not exist: {st.session_state[key_session_state]}")


def add_input(json_data):

    run_info = json_data["run_description"]

    run_str = "RUN INFO: \n\n"
    for k, val in run_info.items():
        run_str += f"{k}: {val} \n\n"
    
    with st.expander("Expand to see RUN INFO"):
        st.info(run_str)

    cols = st.columns(2)
    cont = st.container()
    cont.empty()

    dataset_name = st.selectbox("Select Subset", list(json_data["subset"].keys()), key="subset")

    valid_run_log_files_dict = json_data["log_files"][dataset_name]

    with cols[0]:
        run1_name = st.selectbox("Select Run 1", list(valid_run_log_files_dict.keys()), key="log_dump_run_1")
        log_file_1 = valid_run_log_files_dict[run1_name]
        st.session_state.log_dump_1 = log_file_1

    with cols[1]:   
        run2_name = st.selectbox("Select Run 2", list(valid_run_log_files_dict.keys()), key="log_dump_run_2")
        log_file_2 = valid_run_log_files_dict[run2_name]
        st.session_state.log_dump_2 = log_file_2  

    st.info(f"Log file 1: {st.session_state.log_dump_1}")
    st.info(f"Log file 2: {st.session_state.log_dump_2}")



# def display_nested_cepo_logs(log_data):
#     if isinstance(log_data, dict):
#         for key, value in log_data.items():
#             st.code(value)
#             if not instance(value, (dict, list)):
#                 with st.expander(f"Key: {key}"):
#                     if key != "image_url":
#                         st.code(value)
#                     else:
#                         st.code(f"Image URL here, not displayed for brevity")
#             elif isinstance(value, dict):
#                 display_nested_cepo_logs(value)
#             elif isinstance(value, list):
#                 for item in value:
#                     display_nested_cepo_logs(item)
#     else:
#         st.code(log_data)


def display_sample_data(cont, sample_data, cepo_data=None):
    with cont:
        prompt = sample_data.get("prompt_text", "")
        if len(prompt):
            prompt = prompt.replace(f"Question: {sample_data.get('Question', '')}", "")
        st.write(f"**Prompt:** {prompt}")
        st.write(f"**Question:** {sample_data.get('Question', 'N/A')}")
        st.write(f"**Question Type:** {sample_data.get('Question Type', 'N/A')}")
        st.write(f"**Answer:** {sample_data.get('Answer', 'N/A')}")
        st.write(f"**Model's Output:** {sample_data.get('raw_model_output', 'N/A')}")
        st.write(f"**Prediction:** {sample_data.get('prediction', 'N/A')}")

        if "is_correct" in sample_data:
            is_correct = sample_data["is_correct"]
            accuracy = sample_data.get("accuracy_score", 0.0)
            if is_correct:
                st.success(f"✅ Correct (score: {accuracy:.2f})")
            else:
                st.error(f"❌ Incorrect (score: {accuracy:.2f})")

        if cepo_data is not None:
            st.subheader("CePO Logs")
            with st.expander("Expand"):
                st.write(cepo_data)



def filter_samples():

    p, subset = os.path.split(st.session_state.log_dump_1)
    results_all_run1 = os.path.join(p, f"{os.path.split(p)[1]}_{subset}.json")

    p, subset = os.path.split(st.session_state.log_dump_2)
    results_all_run2 = os.path.join(p, f"{os.path.split(p)[1]}_{subset}.json")
    
    df1 = pd.read_json(results_all_run1)

    df1["sample_file_name"] = df1.apply(lambda row: os.path.join(st.session_state.log_dump_1, f"sample_{row['original_idx']:04d}.json"), axis=1)
    df1["sample_file_exists"] = df1["sample_file_name"].apply(lambda x: os.path.exists(x))
    df1 = df1[df1["sample_file_exists"]]

    df2 = pd.read_json(results_all_run2)
    df2["sample_file_name"] = df2.apply(lambda row: os.path.join(st.session_state.log_dump_2, f"sample_{row['original_idx']:04d}.json"), axis=1)
    df2["sample_file_exists"] = df2["sample_file_name"].apply(lambda x: os.path.exists(x))
    df2 = df2[df2["sample_file_exists"]]

    df_intersect = df1.merge(df2, on="original_idx", suffixes=('_run1', '_run2'))

    return df_intersect


def get_string_query(filter_option, columns=None):
    if filter_option == "All Samples":
        return 'original_idx >= 0' # hacky way of selecting all samples
    elif filter_option == "Run1 Correct":
        return 'accuracy_score_run1 == 1.0'
    elif filter_option == "Run1 Incorrect":
        return 'accuracy_score_run1 == 0.0'
    elif filter_option == "Run2 Correct":
        return 'accuracy_score_run2 == 1.0'
    elif filter_option == "Run2 Incorrect":
        return 'accuracy_score_run2 == 0.0'
    elif filter_option == "Diff Only":
        return 'accuracy_score_run1 != accuracy_score_run2'
    elif filter_option == "Custom Filter":
        custom_text_query = st.text_input("Enter custom filter query (e.g., `accuracy_score_run1 > 0.5 & accuracy_score_run2 < 0.5`):", key="custom_filter_query", placeholder=f"Type something, available_columns = {columns}")
        if custom_text_query:
            return custom_text_query
        else:
            st.warning("No custom filter query provided, defaulting to 'All Samples'")
            return 'original_idx >= 0'
    else:
        st.warning("Unknown filter option selected, defaulting to 'All Samples'")
        return 'original_idx >= 0' # hacky way of selecting all samples
        

def get_cepo_log_data(sample_file_name):

    p, fname = os.path.split(sample_file_name)
    cepo_log_file = os.path.join(p, f"cepo_{fname}")

    cepo_data = None
    if os.path.exists(cepo_log_file):
        with open(cepo_log_file, 'r') as f:
            cepo_data = json.load(f)
    return cepo_data

def main():
    st.set_page_config(page_title=f"Log visualizer", layout="wide")
    st.title(f"Log visualizer")
    st.divider()
    json_data = read_run_json_file_path()
    add_input(json_data)
    filter_option = st.radio(
        "Filters:",
        ["All Samples", "Diff Only", "Run1 Correct", "Run1 Incorrect", "Run2 Correct", "Run2 Incorrect", "Custom Filter"],
        key="filter_option"
    )

    # Check if both log directories are set and get common samples
    if 'log_dump_1' in st.session_state and 'log_dump_2' in st.session_state:
        common_samples = filter_samples()
        with st.expander("Common Samples"):
            st.dataframe(common_samples, use_container_width=True)
        
        if not common_samples.empty:

            string_df_query = get_string_query(st.session_state.filter_option, columns=common_samples.columns.tolist())
            st.info(f"Filtering samples with query: `{string_df_query}`")
            filtered_samples = common_samples.query(string_df_query)

            with st.expander("Filtered Samples"):
                st.dataframe(filtered_samples, use_container_width=True)
            
            # # Update display count
            st.info(f"Showing {len(filtered_samples)} samples based on filter ({len(common_samples)} total common samples)")
            if not filtered_samples.empty:
                st.markdown('##### Sample Comparison')
                num_samples_per_page = st.number_input(f"Num samples per page", value=5, min_value=1, max_value=20)
            
                total_pages = math.ceil(len(filtered_samples) / num_samples_per_page)
                page_numbers = list(range(1, total_pages + 1))
                page = st.selectbox("Page", page_numbers, index=0, key="page_selector")
                
                start_idx = (page - 1) * num_samples_per_page
                end_idx = min(start_idx + num_samples_per_page, len(filtered_samples))
                
                # Display samples for current page
                for i in range(start_idx, end_idx):
                    sample = filtered_samples.iloc[i]
                    
                    with st.expander(f"Sample {sample['original_idx']}"):
                        try:
                            with open(sample['sample_file_name_run1'], 'r') as f1, open(sample['sample_file_name_run2'], 'r') as f2:
                                data1 = json.load(f1)
                                data2 = json.load(f2)

                            cepo_data_1 = get_cepo_log_data(sample['sample_file_name_run1'])
                            cepo_data_2 = get_cepo_log_data(sample['sample_file_name_run2'])

                            # Display image once above both columns
                            if "image_url" in data1 and "image_url" in data2:
                                # Optional: Check if images match
                                images_match = data1["image_url"] == data2["image_url"]
                                if not images_match:
                                    st.warning("Images in the two runs don't match! Showing Run 1's image.")
                                
                                try:
                                    left_co, cent_co,last_co = st.columns(3)
                                    with cent_co:
                                        st.image(data1["image_url"])  
                                except:
                                    st.error("Failed to display image")
                            
                            cols = st.columns(2)
                            
                            # Display first run data
                            display_sample_data(cols[0], data1, cepo_data_1)
                            
                            # Display second run data
                            display_sample_data(cols[1], data2, cepo_data_2)
                        
                        except Exception as e:
                            st.error(f"Error displaying sample {sample['id']}: {str(e)}")
            
                # Navigation buttons
                cols_button = st.columns(3)
                if page > 1:
                    cols_button[0].button("Previous Page", on_click=lambda: st.session_state.update({"page_selector": page-1}))
                cols_button[1].markdown("[Back to Top](#log-visualizer)")
                if page < total_pages:
                    cols_button[2].button("Next Page", on_click=lambda: st.session_state.update({"page_selector": page+1}))
            else:  
                st.warning("No samples match the selected filter criteria.")
        else:
            st.warning("No common samples found between the two runs.")


if __name__ == "__main__":
    main()
