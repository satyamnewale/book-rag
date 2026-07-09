import re
from app.retriever.approch_module import retriever

# naming
strategy = "(similarity + bm25) -> rrf -> reranker"
retriever = retriever()

# -----------------------
# Main
# -----------------------

while True:
    
    file = input("Enter the question name: ")
    if file == "exit":
        break
    else:
        safe_file = re.sub(r'[<>:"/\\|? *]', "_", file)

        result = retriever.invoke(file)

        with open(f"./output/{safe_file}.txt", "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Question: {file}, strategy: {strategy}\n")

            for doc in result:
                f.write(f"\n{doc.page_content}\n")
                f.write("-"*80)
                f.write("\n")
            f.write("\n")
            f.write("-end-"*20)