
import tkinter as tk
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import sys
import os

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = 'OPEN API KEY'

class ChatbotApp:
    def __init__(self, root, directory_path):
        self.root = root
        self.root.title("Endal")

        self.root.geometry("600x400")  # Set the initial size of the window

        self.output_textbox = tk.Text(root, height=10, width=50, state=tk.DISABLED, bg="lavender", wrap=tk.WORD)
        self.output_textbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.input_textbox = tk.Text(root, height=2, width=50, bg="lavender")  # Light gray background
        self.input_textbox.pack(pady=10, padx=10, fill=tk.X)

        self.send_button = tk.Button(root, text="Send", command=self.chat, bg="purple", fg="white")  # Purple button
        self.send_button.pack(pady=10, padx=10, fill=tk.X)

        self.directory_path = directory_path
        self.index = self.construct_index()

    def construct_index(self):
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

        documents = SimpleDirectoryReader(self.directory_path).load_data()

        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        index.save_to_disk('index.json')

        return index

    def chat(self):
        user_input = self.input_textbox.get("1.0", tk.END).strip()
        if user_input.lower() == "exit":
            self.root.quit()
        else:
            conversation_text = self.output_textbox.get("1.0", tk.END) + f"User: {user_input}\n"
            response = self.index.query(conversation_text, response_mode="compact")
            reply = response.response
            conversation_text += f"Endal: {reply}\n"
            self.output_textbox.config(state=tk.NORMAL)
            self.output_textbox.delete("1.0", tk.END)
            self.output_textbox.insert(tk.END, conversation_text)
            self.output_textbox.config(state=tk.DISABLED)
            self.input_textbox.delete("1.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root, "docs")
    root.mainloop()
