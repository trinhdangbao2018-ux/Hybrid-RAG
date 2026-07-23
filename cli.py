import argparse

from src.evaluate import main as run_eval
from src.pipeline import ask, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Local hybrid RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="build the index from documents/")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--show", action="store_true",
                       help="print the retrieved chunks")
    p_ask.add_argument("--no-llm", action="store_true",
                       help="retrieval only, skip generation")
    p_ask.add_argument("--model", default=None, help="ollama model override")

    sub.add_parser("eval", help="run the golden-set evaluation (vector/bm25/hybrid comparison)")

    args = parser.parse_args()
    if args.command == "index":
        build_index()
    elif args.command == "ask":
        answer = ask(args.question, show=args.show,
                     no_llm=args.no_llm, model=args.model)
        if answer:
            print(answer)
    elif args.command == "eval":
        run_eval()


if __name__ == "__main__":
    main()