from flask import Flask, request, render_template
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import logging
import re

# Set up logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
app = Flask(__name__)

# Verify API token
api_token = os.getenv("HF_TOKEN")
if not api_token:
    logging.error("HF_TOKEN not found in .env file")
    raise ValueError("HF_TOKEN not found in .env file")
logging.debug(f"API token loaded: {api_token[:4]}...{api_token[-4:]}")

# Initialize InferenceClient
try:
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=api_token
    )
    logging.debug("InferenceClient initialized successfully with fireworks-ai")
except Exception as e:
    logging.error(f"Error initializing InferenceClient: {str(e)}")
    raise

# Initialize embeddings for RAG
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.debug("Embeddings initialized successfully")
except Exception as e:
    logging.error(f"Error initializing embeddings: {str(e)}")
    raise

# Load and prepare email templates for RAG
try:
    if not os.path.exists("email_templates.txt"):
        logging.error("email_templates.txt not found")
        raise FileNotFoundError("email_templates.txt not found")
    loader = TextLoader("email_templates.txt")
    documents = loader.load()
    if not documents:
        logging.error("No documents loaded from email_templates.txt")
        raise ValueError("No documents loaded from email_templates.txt")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    logging.debug(f"Loaded and split {len(texts)} document chunks")
except Exception as e:
    logging.error(f"Error loading templates: {str(e)}")
    raise

# Create FAISS vector store
try:
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    logging.debug("FAISS vector store and retriever initialized")
except Exception as e:
    logging.error(f"Error creating FAISS vector store: {str(e)}")
    raise

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_prompt = request.form.get("prompt", "").strip()
        recipient = request.form.get("recipient", "").strip()
        logging.info(f"Received prompt: {user_prompt}, recipient: {recipient}")
        
        if not user_prompt or not recipient:
            logging.warning("Empty prompt or recipient provided")
            return render_template("index.html", email_subject="", email_body="Error: Please provide both a prompt and a recipient")
        
        if recipient == user_prompt:
            logging.warning("Recipient matches prompt, likely form error")
            return render_template("index.html", email_subject="", email_body="Error: Recipient cannot be the same as the prompt")
        
        try:
            # Retrieve relevant documents
            docs = retriever.invoke(user_prompt)
            if not docs:
                logging.warning("No documents retrieved")
                return render_template("index.html", email_subject="", email_body="Error: No relevant templates found")
            context = "\n".join([doc.page_content for doc in docs])
            logging.debug(f"Retrieved context: {context}")
            
            # Format prompt for InferenceClient
            prompt = f"""
You are an AI assistant that generates professional emails based on user prompts and relevant email templates.
Use the following context to generate a polite and professional email:

**Context (retrieved templates):**
{context}

**User Prompt:**
{user_prompt}

**Instructions:**
- Generate a concise email with a subject line and body.
- Ensure the tone is professional and appropriate for the recipient (e.g., professor, colleague).
- Address the recipient as "Dear [Recipient]" (use "Professor" for professors).
- Sign off with "Best regards, [Your Name]".
- Return the email in the format:
  Subject: [Subject Line]
  Body: [Email Body]
- Do not include any reasoning or extra tags like <think>.
"""
            # Call InferenceClient
            try:
                completion = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                email_output = completion.choices[0].message.content
                logging.info(f"InferenceClient raw output: {email_output}")
            except Exception as api_error:
                logging.error(f"InferenceClient API error: {str(api_error)}")
                return render_template("index.html", email_subject="", email_body=f"Error: API call failed - {str(api_error)}")
            
            # Remove <think> block if present
            email_output = re.sub(r'<think>.*?</think>\s*', '', email_output, flags=re.DOTALL)
            logging.debug(f"Cleaned output: {email_output}")
            
            # Parse the output
            lines = email_output.strip().split("\n")
            if len(lines) < 2 or not lines[0].startswith("Subject:"):
                logging.warning("Invalid output format after cleaning")
                return render_template("index.html", email_subject="", email_body="Error: Invalid response format from model")
            
            subject = lines[0].replace("Subject: ", "").strip()
            body = "\n".join(lines[1:]).replace("Body: ", "").strip()
            
            # Replace placeholders
            body = body.replace("[Recipient]", recipient)
            body = body.replace("[Your Name]", "Your Name")
            
            logging.info(f"Generated email - Subject: {subject}, Body: {body}")
            return render_template("index.html", email_subject=subject, email_body=body)
        except Exception as e:
            logging.error(f"Error during email generation: {str(e)}")
            return render_template("index.html", email_subject="", email_body=f"Error: {str(e)}")
    
    return render_template("index.html", email_subject="", email_body="")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)