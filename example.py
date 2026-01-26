from src.models.logic_rag import LogicRAG

# Initialize RAG system
rag = LogicRAG('dataset/musique_corpus.json')
rag.set_max_rounds(5)
rag.set_top_k(3)

# Ask a question
answer, contexts, rounds = rag.answer_question("What is the capital of France?")
print(f"Answer: {answer}")
print(f"Retrieved in {rounds} rounds")