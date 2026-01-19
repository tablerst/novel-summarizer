from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def book_hash(normalized_text: str) -> str:
    return sha256_text(normalized_text)


def chapter_hash(book_hash_value: str, title: str, text: str) -> str:
    return sha256_text(f"{book_hash_value}::{title}::{text}")


def chunk_hash(chapter_hash_value: str, text: str, split_params: str) -> str:
    return sha256_text(f"{chapter_hash_value}::{split_params}::{text}")
