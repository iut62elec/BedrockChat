# BedrockChat

BedrockChat acts as a conversational interface, leveraging generative AI models fine-tuned on your content. This feature provides users with accurate, timely information and expert insights, transforming static content consumption into a dynamic, interactive experience. BedrockChat is powered by AWS Bedrock, AWS Kendra, LangChain and Streamlit for UI. 

Potential benefits of BedrockChat include an increase in user numbers and subscriptions for your website, which could lead to an increase in revenue. Furthermore, it could help solidify your position as a trusted information source within the industry.

This AI-driven application offers:
* State-of-the-art Retrieval Augmented Generation (RAG) mechanism designed to mitigate hallucinations and model drifting.
* The ability to reference sources, allowing users to click and access the full content of your website, reinforcing their engagement.
* A modular architecture, allowing for flexibility in swapping the LLM or knowledge base with newer technologies as they become available.
* Ensured data security as you host and deploy your retrieval system and LLM in your account, eliminating the need to send data to a third party.

## Installation

To set up the application, follow the steps below:

**Prerequisites**
You need to have AWS account with access to AWS Bedrock and AWS Kendra. Create an index in AWS Kendra and using one of the provided connectors (for example, Web crawler), ingest documents into the AWS Kendra index.

1. Clone the repository:
    ```bash
    git clone https://github.com/iut62elec/BedrockChat.git
    ```
2. Navigate into the project directory:
    ```bash
    cd BedrockChat
    ```
3. Create and activate a virtual environment:
    ```bash
    python3.8 -m venv .venv
    source .venv/bin/activate
    ```
4. Upgrade pip and install the required dependencies:
    ```bash
    pip install --upgrade pip
    pip install boto3==1.26.163
    pip install watchdog
    pip install streamlit
    pip install langchain
    pip install streamlit-cognito-auth
    python -m pip install ./whl_files/botocore-1.29.162-py3-none-any.whl
    python -m pip install ./whl_files/boto3-1.26.162-py3-none-any.whl
    python -m pip install ./whl_files/awscli-1.27.162-py3-none-any.whl
    ```
5. Deactivate and reactivate the virtual environment with AWS credentials (Remember to replace "AWS_PROFILE_WITH_Bedrock_KENDRA_access" and "XXXXX" with your actual AWS profile and Kendra index ID):
    ```bash
    deactivate
    export AWS_PROFILE=AWS_PROFILE_WITH_Bedrock_KENDRA_access
    export AWS_REGION="us-east-1" 
    export KENDRA_INDEX_ID_BR="XXXXX"
    source .venv/bin/activate

    ```
6. Navigate into the source code and run the application:
    ```bash
    cd ./src
    streamlit run app_BedrockChat.py bedrock
    ```

This will open a local server at http://localhost:8501/, where you can start interacting with the AI!

## Disclaimer

This repository represents my viewpoints and not those of my employer, Amazon Web Services (AWS). All product names, logos, and brands are the property of their respective owners. This code acts as a sample to use AWS Bedrock for Chatbot and has been tested on a Mac device and works well, but use at your own caution. Also note that AWS Kendra and AWS Bedrock are not part of AWS free tier.
