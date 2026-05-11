import os
from typing import List
from pathlib import Path


def load_text_file(file_path: str) -> str:
    """加载单个文本文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_all_documents(data_dir: str) -> List[dict]:
    """加载data_dir下所有文本文档，返回 [{filename, content}] 列表"""
    docs = []
    data_path = Path(data_dir)
    for file_path in sorted(data_path.glob("*.txt")):
        content = load_text_file(str(file_path))
        docs.append({
            "filename": file_path.name,
            "content": content,
            "path": str(file_path)
        })
        print(f"  已加载: {file_path.name} ({len(content)} 字符)")
    return docs


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """将长文本按段落切分为chunks，尽量保持段落完整"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)
        if current_size + para_size > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # 保留最后一个段落作为overlap
            if overlap > 0 and current_chunk:
                current_chunk = [current_chunk[-1]]
                current_size = len(current_chunk[0])
            else:
                current_chunk = []
                current_size = 0
        current_chunk.append(para)
        current_size += para_size

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
