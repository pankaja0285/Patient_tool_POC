import streamlit as st
import time

from pt_scripts import *
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Function to apply custom CSS for styling the tab edge
def set_tab_style():
    # Add custom CSS for styling the tabs
    st.markdown("""
    <style>
    /* Remove gap between tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    /* Style for inactive tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6; /* Light grey background */
        border-top-right-radius: 10px; /* Rounded top right edge */
    }

    /* Style for focused/active tabs */
    .stTabs [aria-selected="true"] {
        background-color: #90EE90; /* Light green background for focused tab */
        color: darkgreen;
        font-weight: bold;
        background-top-right-radius: 15px;
    }

    /* Style for active tabs */
    .stTabs [data-baseweb="tab"] {
        width: 200px;
        height: 60px; /* Adjust height if needed */
        display: flex; /* Use flexbox for vertical centering */
        align-items: center; /* Vertically center content */
        justify-content: center; /* Horizontally center content */    
        padding-top: 10px;
        padding-bottom: 10px;    
    }
    /* Optional: Apply focus style (e.g., thicker bottom border) */
    .stTabs [role=tab]:focus {
        box-shadow: none; /* Remove default focus box shadow */
        outline: none; /* Remove default outline */
        border-bottom: 3px solid blue; /* Example focus style */
    }
    </style>
    """, unsafe_allow_html=True)

# Function to handle button click
def download_action(client=None, model_name=""):
    df = None
    # st.session_state.button_clicked = True
    # st.session_state.button_disabled = True  # Disable the button

    # Perform your action here (e.g., process data, display results)
    d_file = f"./data/clinical_notes_sample.csv"
    r_file = f"./data/medical_transcriptions_raw.csv"
    rows_to_download = int(entry_label)
    print(f"Rows to download: {rows_to_download}")
    # Create a sample DataFrame
    msg = "Downloading/formatting patient notes data. Please wait..."
    # info_placeholder.info("Downloading/formatting patient notes data. Please wait...")
    # call to download and set content
    df = download_data_and_set_content(client=client, model_name=model_name,
                                    data_file=d_file, raw_file=r_file,
                                    update_to_existing=True, 
                                    default_download=rows_to_download)

    if not df is None:
        st.write("### Response Data:")
        # Display DataFrame with fixed header in a scrollable area
        st.dataframe(df.style.set_properties(**{'white-space': 'nowrap', 'overflow-x': 'auto'}),
                    height=100, use_container_width=True)

    # Reset button
    time.sleep(2)
    # st.session_state.button_clicked = False
    # st.session_state.sidebar_messages.append("Processing complete. Button re-enabled.")
    # st.sidebar.success("Request processed successfully!") # Display success message in the sidebar

    # Set to show 2nd tab
    st.session_state.show_second_tab_content = True

# Apply the custom styles
set_tab_style()

# Create the two tabs
tab1, tab2 = st.tabs(["Download/Format Data", "Prompt/Response"])

# Other setup
model_name = "gemini-2.0-flash-001"

# create a genai client
client = genai.Client()
print(f"client: {client}")
# Initialize session state for the flag
if 'show_second_tab_content' not in st.session_state:
    st.session_state.show_second_tab_content = False
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# --- Custom Button Styling ---
# Define CSS for the button based on its state
button_style = """
<style>
div.stButton > button {
    background-color: #4CAF50; /* Green background */
    color: white; /* White text */
    padding: 2px 8px;
    margin: 1px 0;
    border: none;
    cursor: pointer;
    width: 225px;
}

</style>
"""

# --- Tab 1 Content ---
with tab1:
    st.header("Download/Format Data")

    # Label and Text entry
    entry_label = st.text_input("Enter number of rows to download:", key="entry_input", value=15)
    st.markdown(button_style, unsafe_allow_html=True)
    download_button = st.button("Download and Apply", key="show_data_button")
    if download_button:
        if entry_label and (not st.session_state.show_second_tab_content):
            with st.spinner("Download/Formatting data, please wait..."):
                time.sleep(2) # Simulate a delay
                download_action(client=client, model_name=model_name)
                st.info("Download/Format data completed.")
        else:
            warning_msg = "Please enter a value before clicking 'Show Data'."
            st.warning(warning_msg)       
        
# --- Tab 2 Content ---
with tab2:
    st.header("Prompt/Response")

    # Label and Text entry
    prompt = st.text_input("Enter your prompt:", key="prompt_input")

    # Check if text is entered and the user pressed Enter
    # Streamlit automatically re-runs when text input changes,
    # so we can check if label_2 has a value to trigger the response.
    if prompt:
        if st.session_state.show_second_tab_content:
            data_file = "./data/clinical_notes_sample.csv"
            df_rag_data = prep_rag_data(data_file=data_file, client=client)
            st.session_state.data_loaded = True

            if st.session_state.data_loaded:
                # Initialize a state variable to store the response if not already present
                if "response" not in st.session_state:
                    st.session_state.response = ""

                # Check if the Enter key was pressed (Streamlit handles this by rerunning the app)
                # We can detect this by checking if the prompt value has changed
                if st.session_state.prompt_input != st.session_state.get("last_prompt", ""):
                    # Simulate fetching a response
                    with st.spinner("Fetching response..."):
                        time.sleep(2) # Simulate a delay
                        # st.session_state.response = f"This is a simulated response to your prompt: {st.session_state.prompt_input}"
                        qry = st.session_state.prompt_input
                        resp = doctor_chatbot(qry, df=df_rag_data, client=client, genai_model_name=model_name)
                        st.session_state.response = resp
                    # Update the last prompt value in session state to avoid fetching again on every rerun
                    st.session_state.last_prompt = st.session_state.prompt_input

                # Display the response in a scrollable text area
                st.text_area("Response:", st.session_state.response, height=200) # Using st.text_area for scrollable output
                # ask follow-up question
                st.write("Feel free to ask another question!") # Follow-up label
            else:
                st.session_state.data_loaded = False
                warning_msg = "Loading prepped data failed!"
                st.info(warning_msg)
        else:
                warning_msg = "Content for Tab 2 is hidden. Finish Download data and Display - tab actions."
                st.info(warning_msg)
     
