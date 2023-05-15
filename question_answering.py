from apis.qa_chain import QAChain

def main():
    print("Loading QA Chain...")
    qa_chain = QAChain()
    print("...Done. Ready to chat!")

    while True:
        query = input("Type a question and press enter: ")
        answer = qa_chain.qa(query)
        print(answer)

if __name__ == "__main__":
    main()