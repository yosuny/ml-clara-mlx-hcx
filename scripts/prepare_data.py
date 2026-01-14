import os
import glob
from pypdf import PdfReader
import json

def extract_text_from_pdfs(data_dir, output_file):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    all_text = []
    
    print(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            # Simple cleaning
            # Remove excessive newlines or page numbers if easy, 
            # but for CLaRa raw text is usually fine if tokenized.
            all_text.append(text)
            print(f"Extracted {len(text)} characters.")
            
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")

    # Save to JSONL or TXT
    # We'll save as a single text file for simplicity in this MVP
    # Separate documents by some delimiter if needed, or just concat.
    # CLaRa context compression usually works on document level. 
    # Let's save as JSONL where each line is a document.
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_text:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"Saved extracted data to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="knowledge_data")
    parser.add_argument("--output_file", type=str, default="data/train_data.jsonl")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    extract_text_from_pdfs(args.data_dir, args.output_file)
