"""Command line entry point: build the corpus, or query one arm by hand."""

import argparse

from dotenv import load_dotenv

from ncert_rag.core.registry import BOOKS, BY_SLUG
from ncert_rag.ingest.pipeline import build
from ncert_rag.retrieve import ARMS
from ncert_rag.store import db


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="ncert-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build", help="download, parse and index the books")
    builder.add_argument("--rebuild", action="store_true", help="ignore hashes")
    builder.add_argument("--book", action="append", help="limit to these slugs")

    search = sub.add_parser("search", help="run one query against one arm")
    search.add_argument("query")
    search.add_argument("--arm", default="bm25", choices=sorted(ARMS))
    search.add_argument("-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        if args.book:
            invalid = [slug for slug in args.book if slug not in BY_SLUG]
            if invalid:
                parser.error(
                    f"Unknown book slug(s): {', '.join(invalid)}. "
                    f"Valid slugs: {', '.join(sorted(BY_SLUG))}"
                )
            books = [BY_SLUG[slug] for slug in args.book]
        else:
            books = list(BOOKS)
        build(books, rebuild=args.rebuild)
        return

    conn = db.connect()
    for hit in ARMS[args.arm](conn).retrieve(args.query, args.k):
        where = f"{hit.book} ch{hit.chapter}"
        if hit.section:
            where += f" §{hit.section}"
        print(f"[{hit.score:6.3f}] {where} p{hit.page}\n    {hit.text[:200]}\n")
