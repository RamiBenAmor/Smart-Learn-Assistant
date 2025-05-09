# 📚 SmartLearn Assistant

**SmartLearn Assistant** is an AI-powered application designed to revolutionize the way students learn. Leveraging cutting-edge Natural Language Processing (NLP) technologies, it offers a smart chatbot that interacts fluently with users, answers questions about course content, provides detailed explanations, and generates interactive quizzes tailored to each student's level.

## 🚀 Features

- 🧠 **AI Chatbot:** Real-time conversation with a course-aware assistant.
- 🔍 **Smart Content Search:** Automatically identifies and returns the most relevant content from your uploaded course material.
- ✂️ **Two Types of Content Fragmentation:**
  - **Standard Search:** Splits content into 512-token chunks for general NLP processing.
  - **Advanced Search:** Splits content based on structured sections (titles, headers) of the course.
- 📝 **Adaptive Quiz Generation:** Generates quizzes based on your level with increasing difficulty.
- 📄 **Automatic Summarization:** Produces concise summaries of each course section for quick review.

## 🧰 Tech Stack

- `Flask` – Web application framework  
- `MySQL` – User authentication and data storage  
- `Transformers` & `CTransformers` – LLM integration for inference  
- `FAISS` – Semantic vector search  
- `Torch` – Deep learning backend  
- `NLTK` – Tokenization and text preprocessing utilities  

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smartlearn-assistant.git
   cd smartlearn-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the LLaMA model**

   Download `llama-2-7b-chat.Q4_K_M.gguf` from Hugging Face:  
   👉 [Download Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf)

4. **Place the model**

   Put the downloaded `.gguf` model in the directory used by your `ctransformers` loader.

## 🧪 How It Works

- Upload a **PDF** course file using the UI.
- Choose one of the two **fragmentation strategies**:
  - **Standard Search:** Chunks of 512 tokens.
  - **Advanced Search:** Based on the course's structured titles/sections.
- The assistant will:
  - Extract and clean the text
  - Generate text embeddings
  - Store them in a FAISS index
  - Use LLaMA to answer questions intelligently
- Additional capabilities:
  - Generate personalized quizzes
  - Produce instant summaries of key content

## 📸 UI Pages

- `page1.html` – Landing page  
- `accueil.html` – Home dashboard  
- `interface.html` – Upload interface  
- `index.html` – Results and interaction  

## 🔐 Authentication

SmartLearn Assistant includes a simple authentication system using **Flask** and **MySQL** to manage user registration and login. Users can register with their email, username, and phone number, and log in using their email and password.

### Registration:
- **Checks if the email or username already exists** in the database.
- **Hashing** the password using `werkzeug.security.generate_password_hash` for secure storage.
- **Stores user details** in MySQL.

### Login:
- **Verifies user credentials** by checking if the email exists and if the password matches the hashed value.
- **Sets up a session** for the logged-in user.

## 📬 Contact

For questions, ideas, or issues, feel free to open an issue or contact the developers.

---

**SmartLearn Assistant** makes learning faster, deeper, and smarter. 💡
