import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

class TextPreprocessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Preprocessing Tool")
        self.root.geometry("1200x800")  # Increased window size for better layout
        self.root.configure(bg='#000000')  # Black background
        
        # Configure simple black and red color scheme
        self.colors = {
            'primary': '#000000',  # Black
            'secondary': '#1a1a1a',  # Dark gray
            'accent_red': '#ff0000',  # Red
            'text_white': '#ffffff',  # White
            'text_red': '#ff0000',   # Red
        }
        
        # Configure styles
        self.configure_styles()
        
        # Download required NLTK data
        self.download_nltk_data()
        
        self.setup_ui()
        
    def configure_styles(self):
        """Configure simple black and red styles"""
        style = ttk.Style()
        
        # Configure theme
        style.theme_use('clam')
        
        # Configure colors for different widgets
        style.configure('Primary.TFrame', background=self.colors['primary'])
        style.configure('Secondary.TFrame', background=self.colors['secondary'])
        
        # Heading style - Red text on black
        style.configure('RedHeading.TLabel', 
                       background=self.colors['primary'],
                       foreground=self.colors['accent_red'],
                       font=('Arial', 24, 'bold'),
                       padding=10)
        
        # Button styles - Red text on dark background
        style.configure('RedText.TButton',
                       background=self.colors['secondary'],
                       foreground=self.colors['accent_red'],
                       font=('Arial', 11, 'bold'),
                       borderwidth=2)
        
        style.map('RedText.TButton',
                 background=[('active', self.colors['primary']),
                           ('pressed', self.colors['primary'])])
        
        # Label styles
        style.configure('White.TLabel',
                       background=self.colors['secondary'],
                       foreground=self.colors['text_white'],
                       font=('Arial', 15, 'bold'))
        
        style.configure('Red.TLabel',
                       background=self.colors['secondary'],
                       foreground=self.colors['text_red'],
                       font=('Arial', 15, 'bold'))
        
        # Notebook style
        style.configure('Black.TNotebook',
                       background=self.colors['primary'],
                       tabmargins=[2, 5, 2, 0])
        
        style.configure('Black.TNotebook.Tab',
                       background=self.colors['secondary'],
                       foreground=self.colors['text_white'],
                       padding=[20, 10],
                       font=('Arial', 10, 'bold'))
        
        style.map('Black.TNotebook.Tab',
                 background=[('selected', self.colors['primary'])],
                 foreground=[('selected', self.colors['accent_red'])])
    
    def download_nltk_data(self):
        """Download required NLTK datasets"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, style='Primary.TFrame', padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for proper expansion
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)  # Input section - don't expand
        main_frame.rowconfigure(3, weight=1)  # Notebook - expand to fill space
        
        # Heading with red text on black background
        heading_frame = ttk.Frame(main_frame, style='Primary.TFrame')
        heading_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        heading_frame.columnconfigure(0, weight=1)
        
        heading_label = ttk.Label(heading_frame, 
                                 text="🔥 TEXT PREPROCESSING TOOL 🔥", 
                                 style='RedHeading.TLabel')
        heading_label.grid(row=0, column=0)
        
        # Input section with narrower text box
        input_frame = ttk.Frame(main_frame, style='Secondary.TFrame', padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Label(input_frame, text="Enter Text:", style='Red.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Narrower input text box (width reduced from 80 to 40)
        self.text_input = scrolledtext.ScrolledText(input_frame, 
                                                   height=8,
                                                   width=40,  # Reduced width by half
                                                   font=('Arial', 12),
                                                   bg=self.colors['secondary'],
                                                   fg=self.colors['text_white'],
                                                   insertbackground=self.colors['text_white'],
                                                   selectbackground=self.colors['accent_red'],
                                                   relief=tk.FLAT,
                                                   bd=2)
        self.text_input.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Center the input box
        input_frame.columnconfigure(0, weight=1)
        
        # Process button with red text
        button_frame = ttk.Frame(main_frame, style='Primary.TFrame')
        button_frame.grid(row=2, column=0, pady=(10, 20))
        button_frame.columnconfigure(0, weight=1)
        
        self.process_btn = ttk.Button(button_frame, 
                                     text="🔥 PROCESS TEXT 🔥", 
                                     command=self.process_text,
                                     style='RedText.TButton')
        self.process_btn.grid(row=0, column=0, ipadx=20, ipady=8)
        
        # Create notebook for tabs with proper expansion
        notebook_frame = ttk.Frame(main_frame, style='Primary.TFrame')
        notebook_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        notebook_frame.columnconfigure(0, weight=1)
        notebook_frame.rowconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(notebook_frame, style='Black.TNotebook')
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.create_tokens_tab()
        self.create_bow_tab()
        self.create_tfidf_tab()
        self.create_pos_tab()
        
        # Status bar at the bottom
        status_frame = ttk.Frame(main_frame, style='Secondary.TFrame')
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("🔥 Ready to process text...")
        status_bar = ttk.Label(status_frame, 
                              textvariable=self.status_var, 
                              relief=tk.SUNKEN, 
                              anchor=tk.W,
                              style='White.TLabel',
                              padding=8)
        status_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        status_frame.columnconfigure(0, weight=1)
    
    def create_styled_text_widget(self, parent, height=8, width=80):
        """Create a consistently styled text widget with increased width"""
        return scrolledtext.ScrolledText(parent, 
                                       height=height, 
                                       width=width,  # Increased from 40 to 80
                                       font=('Consolas', 11),
                                       bg=self.colors['secondary'],
                                       fg=self.colors['text_white'],
                                       insertbackground=self.colors['text_white'],
                                       selectbackground=self.colors['accent_red'])
    
    def create_tokens_tab(self):
        """Create tab for tokenization results with wider boxes"""
        tab = ttk.Frame(self.notebook, style='Secondary.TFrame', padding="10")
        self.notebook.add(tab, text="TOKENS")
        
        # Configure grid for better layout
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)
        
        # Original tokens - wider box
        ttk.Label(tab, text="Original Tokens:", style='Red.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(5, 8))
        self.original_tokens_text = self.create_styled_text_widget(tab, width=60)
        self.original_tokens_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        # Stemmed tokens - wider box
        ttk.Label(tab, text="Stemmed Tokens:", style='Red.TLabel').grid(row=0, column=1, sticky=tk.W, pady=(5, 8))
        self.stemmed_tokens_text = self.create_styled_text_widget(tab, width=60)
        self.stemmed_tokens_text.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(0, 10))
        
        # Lemmatized tokens - full width box
        ttk.Label(tab, text="Lemmatized Tokens:", style='Red.TLabel').grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 8))
        self.lemmatized_tokens_text = self.create_styled_text_widget(tab, width=100)
        self.lemmatized_tokens_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    
    def create_bow_tab(self):
        """Create tab for Bag of Words results with wider boxes"""
        tab = ttk.Frame(self.notebook, style='Secondary.TFrame', padding="10")
        self.notebook.add(tab, text="BAG OF WORDS")
        
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)
        
        # Vocabulary - wider box
        ttk.Label(tab, text="Vocabulary:", style='Red.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(5, 8))
        self.bow_vocab_text = self.create_styled_text_widget(tab, height=6, width=100)
        self.bow_vocab_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        # BOW Matrix - wider box
        ttk.Label(tab, text="BOW Matrix:", style='Red.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(5, 8))
        self.bow_matrix_text = self.create_styled_text_widget(tab, height=6, width=100)
        self.bow_matrix_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    
    def create_tfidf_tab(self):
        """Create tab for TF-IDF results with wider boxes"""
        tab = ttk.Frame(self.notebook, style='Secondary.TFrame', padding="10")
        self.notebook.add(tab, text="TF-IDF")
        
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)
        
        # Vocabulary - wider box
        ttk.Label(tab, text="Vocabulary:", style='Red.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(5, 8))
        self.tfidf_vocab_text = self.create_styled_text_widget(tab, height=6, width=100)
        self.tfidf_vocab_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        # TF-IDF Matrix - wider box
        ttk.Label(tab, text="TF-IDF Matrix:", style='Red.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(5, 8))
        self.tfidf_matrix_text = self.create_styled_text_widget(tab, height=6, width=100)
        self.tfidf_matrix_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    
    def create_pos_tab(self):
        """Create tab for POS tagging results with wider box"""
        tab = ttk.Frame(self.notebook, style='Secondary.TFrame', padding="10")
        self.notebook.add(tab, text="POS TAGS")
        
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        
        ttk.Label(tab, text="Part-of-Speech Tags:", style='Red.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(5, 8))
        self.pos_tags_text = self.create_styled_text_widget(tab, height=15, width=100)
        self.pos_tags_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    
    def preprocess_text(self, text):
        """Your original preprocessing function"""
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens]

        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

        pos_tags = pos_tag(lemmatized_tokens)

        vectorizer_bow = CountVectorizer()
        X_bow = vectorizer_bow.fit_transform([" ".join(lemmatized_tokens)])
        vocab_bow = vectorizer_bow.get_feature_names_out()
        bow = X_bow.toarray()

        vectorizer_tfidf = TfidfVectorizer()
        X_tfidf = vectorizer_tfidf.fit_transform([" ".join(lemmatized_tokens)])
        vocab_tfidf = vectorizer_tfidf.get_feature_names_out()
        tfidf_array = X_tfidf.toarray()

        return {
            "original_tokens": tokens,
            "stemmed": stemmed_tokens,
            "lemmatized": lemmatized_tokens,
            "pos_tags": pos_tags,
            "bow_vocab": vocab_bow,
            "bow_matrix": bow,
            "tfidf_vocab": vocab_tfidf,
            "tfidf_matrix": tfidf_array
        }
    
    def process_text(self):
        """Process the text and update the GUI with results"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to process.")
            return
        
        try:
            self.status_var.set("🔥 Processing your text...")
            self.process_btn.config(state='disabled')
            self.root.update()
            
            result = self.preprocess_text(text)
            
            # Update tokens tab
            self.original_tokens_text.delete("1.0", tk.END)
            self.original_tokens_text.insert("1.0", ", ".join(result["original_tokens"]))
            
            self.stemmed_tokens_text.delete("1.0", tk.END)
            self.stemmed_tokens_text.insert("1.0", ", ".join(result["stemmed"]))
            
            self.lemmatized_tokens_text.delete("1.0", tk.END)
            self.lemmatized_tokens_text.insert("1.0", ", ".join(result["lemmatized"]))
            
            # Update BOW tab
            self.bow_vocab_text.delete("1.0", tk.END)
            self.bow_vocab_text.insert("1.0", ", ".join(result["bow_vocab"]))
            
            self.bow_matrix_text.delete("1.0", tk.END)
            self.bow_matrix_text.insert("1.0", str(result["bow_matrix"]))
            
            # Update TF-IDF tab
            self.tfidf_vocab_text.delete("1.0", tk.END)
            self.tfidf_vocab_text.insert("1.0", ", ".join(result["tfidf_vocab"]))
            
            self.tfidf_matrix_text.delete("1.0", tk.END)
            self.tfidf_matrix_text.insert("1.0", str(result["tfidf_matrix"]))
            
            # Update POS tags tab
            self.pos_tags_text.delete("1.0", tk.END)
            pos_tags_str = "\n".join([f"{word}: {tag}" for word, tag in result["pos_tags"]])
            self.pos_tags_text.insert("1.0", pos_tags_str)
            
            self.status_var.set("✅ Processing completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing:\n{str(e)}")
            self.status_var.set("❌ Error occurred")
        finally:
            self.process_btn.config(state='normal')

def main():
    root = tk.Tk()
    app = TextPreprocessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()