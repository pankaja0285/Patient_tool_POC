# Patient tool POC
## Tool mainly to serve as a learning playground for genai

## Features

- Download data and assemble in the format desired with google api and gemini model
- save the assembled data
- Create rag text and embeddings
- Use google api and gemini model and get the response to user queries



## Tech



- [streamlit] - to demo as a web app!
- [VSC] - visual studio code as a code development environment
- [google api] - genai google api

## Folder structure
| Patient_tool_POC |</br>
| ---|data|</br>
| ---|data|clinical_notes_sample.csv|</br>
| ---|data|clinical_notes_sample_baseline.csv|</br>
| ---|data|medical_transcriptions_raw.csv|</br>
| ---|.env|</br>
| ---|.gitignore|</br>
| ---|app.py|</br>
| ---|pt_scripts.py|</br>
| ---|requirements.txt|</br>

NOTE: about .env - this file contains the Google api key pertaining to your own account, hence not checked in 
and is part of .gitignore so it ignores during any add/commit/push, so there's no misuse of PID.

## Installation
Update pip
```
python -m pip install --upgrade pip
```
Create a virtual environment
```
python -m venv pt_venv
```
Install the dependencies
```
pip install -r requirements.txt
```

## Instructions
Instructions on how to use them in your own application are linked below.
Open your favorite Terminal and run these commands.
```
streamlit run app.py
```
First Tab:

```
Click on Download data button and this downloads a default of 15. 
```

Second Tab:

```
Here enter a text and hit enter to get a response from the genai call behind the scenes
```
## Conclusion

Room for improvement and expanding is beyond the scope of this POC, but one can take this and do the same and explore to any level desired.
