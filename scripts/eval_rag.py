import mlx.core as mx
from mlx_lm import load, generate
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_pdf_text(pdf_text_file):
    """Load extracted PDF text."""
    with open(pdf_text_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['text']

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    return chunks

def retrieve_top_k(question, chunks, k=3):
    """Simple TF-IDF based retrieval."""
    # Create corpus
    corpus = chunks + [question]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Question is the last item
    question_vec = tfidf_matrix[-1]
    chunk_vecs = tfidf_matrix[:-1]
    
    # Compute cosine similarity
    similarities = cosine_similarity(question_vec, chunk_vecs)[0]
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return [chunks[i] for i in top_k_indices]

def simple_rag(question, pdf_text, model, tokenizer, max_tokens=128):
    """Simple RAG: retrieve + generate."""
    # 1. Chunk PDF text
    chunks = chunk_text(pdf_text, chunk_size=512, overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    # 2. Retrieve top-k relevant chunks
    relevant_chunks = retrieve_top_k(question, chunks, k=3)
    
    # 3. Construct prompt
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Context:
{context}

Question: {question}
Answer:"""
    
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
    
    # 4. Generate answer
    answer = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Evaluate original model with RAG")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--pdf_text", type=str, default="data/train_data.jsonl", help="Path to extracted PDF text")
    parser.add_argument("--questions", type=str, default="data/eval_questions.jsonl", help="Path to evaluation questions")
    parser.add_argument("--output", type=str, default="results/rag_answers.jsonl", help="Output file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)
    
    print(f"Loading PDF text from {args.pdf_text}")
    pdf_text = ""
    with open(args.pdf_text, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            pdf_text += item['text'] + "\n"
    
    print(f"Loading questions from {args.questions}")
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Evaluating {len(questions)} questions with RAG...")
    
    results = []
    for i, q_data in enumerate(questions, 1):
        question = q_data['question']
        print(f"\n[{i}/{len(questions)}] {question}")
        
        try:
            answer = simple_rag(
                question, 
                pdf_text, 
                model, 
                tokenizer,
                max_tokens=args.max_tokens
            )
            
            result = {
                **q_data,
                "rag_answer": answer,
                "status": "success"
            }
            print(f"Answer: {answer[:100]}...")
            
        except Exception as e:
            print(f"Error: {e}")
            result = {
                **q_data,
                "rag_answer": "",
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to {args.output}")
    print(f"Successfully answered {sum(1 for r in results if r['status'] == 'success')}/{len(results)} questions")

if __name__ == "__main__":
    main()
