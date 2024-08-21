# Group Chat Story Generator


## Project Description
The Group Chat Story Generator is a Python-based project designed to automatically generate stories or summaries from group chat data. Using state-of-the-art natural language processing (NLP) models, specifically the Pegasus model from the Hugging Face transformers library, this tool can process a CSV file containing group chat messages and transform the conversational data into coherent, summarized text. This project is ideal for creating narrative stories or concise summaries from lengthy chat logs.

## Features
- **Data Preprocessing**: The project includes scripts for preprocessing group chat data, including cleaning, formatting, and preparing it for input to the Pegasus model.
- **Model Fine-Tuning**: Utilizes the Hugging Face Transformers library to fine-tune the Pegasus model on custom group chat data.
- **Story Generation**: Once the model is trained, it can generate stories based on input group chat conversations.
- **Evaluation**: Provides methods for evaluating the generated stories, including metrics such as coherence, fluency, and relevance.

## Getting Started on Google Colab 
1. **Open the Notebook in Google Colab**
     You can run the project directly in Google Colab by opening the notebook from your GitHub repository.If the notebook is stored locally, upload it to Colab.
   
2.**Install Required Dependencies**
    Run the following command in a Colab code cell to install the necessary libraries:
   ```bash
  !pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
  ```

3.**Mount Google Drive (Optional)**
    If your group chat data is stored on Google Drive, mount your Drive in Colab:
  ```bash
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  
4.**Upload Your Data**
    You can either upload your group chat CSV file directly to the Colab environment or access     it from Google Drive.
  
5.**Run the Notebook**
  Follow the instructions in the notebook to load your data, configure the Pegasus model, and     generate summaries or stories.


## Visual Assets
![Input Interface](https://github.com/Aakash2003jain/GroupChatStoryGeneration/assets/102961260/b5b9eb7c-5eed-43f5-83ff-45101bf8ad2c)
*Input Interface: A visual depiction showcasing the user interface of the GroupChatStoryGen webpage.*

![Screenshot 2024-05-05 215831](https://github.com/Aakash2003jain/GroupChatStoryGeneration/assets/102961260/dbadaa15-441b-4a60-b458-42b98be3a60f)
*Output Interface: A visual representation illustrating the output generated by the GroupChatStoryGen model.*


## Acknowledgements
- This project utilizes the Google's Pegasus model.
- Thanks to the Hugging Face team for their excellent Transformers library.



