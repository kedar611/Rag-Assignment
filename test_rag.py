import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Ensure project path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from rag_engine import RAGEngine, DocumentChunk
import faiss
import numpy as np

def create_sample_pdf(pdf_path: str):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    # Page 1
    c.drawString(72, 750, "Project Apollo: Next-Generation Renewable Energy Architecture")
    c.drawString(72, 720, "1. Executive Summary")
    c.drawString(72, 690, "Project Apollo is an initiative designed to transition corporate data centers to 100% solar and wind power by 2030.")
    c.drawString(72, 660, "The total capital expenditure allocated for this initiative is $45 million over five fiscal years.")
    c.showPage()
    
    # Page 2
    c.drawString(72, 750, "2. Technical Specifications and Battery Storage")
    c.drawString(72, 720, "Apollo uses high-density lithium-iron-phosphate (LFP) battery banks providing 250 MWh of storage capacity.")
    c.drawString(72, 690, "Thermal management is governed by closed-loop dielectric liquid cooling.")
    c.drawString(72, 660, "Key risk factor: Battery cell supply chain constraints in Q3.")
    c.showPage()
    c.save()
    print(f"[OK] Sample PDF created at {pdf_path}")

def test_extraction_and_chunking(pdf_path: str):
    engine = RAGEngine(provider="gemini", api_key=None, chunk_size=300, chunk_overlap=50)
    with open(pdf_path, "rb") as f:
        pages_data = engine.extract_text_from_pdf(f.read())
    
    assert len(pages_data) == 2, f"Expected 2 pages, got {len(pages_data)}"
    print(f"[OK] Extracted {len(pages_data)} pages successfully.")
    
    chunks = engine.chunk_pdf(pages_data)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    print(f"[OK] Generated {len(chunks)} chunks.")
    for c in chunks:
        print(f"  - Chunk {c.chunk_id} (Page {c.page_number}): {c.content[:60]}...")

def test_faiss_indexing():
    # Verify FAISS indexing with simulated embeddings
    dimension = 768
    num_items = 4
    np.random.seed(42)
    fake_vectors = np.random.randn(num_items, dimension).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(fake_vectors, axis=1, keepdims=True)
    fake_vectors /= norms

    index = faiss.IndexFlatIP(dimension)
    index.add(fake_vectors)
    assert index.ntotal == num_items

    # Query with first vector
    query_vec = fake_vectors[0:1]
    scores, indices = index.search(query_vec, k=2)
    assert indices[0][0] == 0, f"Expected closest match to be index 0, got {indices[0][0]}"
    assert abs(scores[0][0] - 1.0) < 1e-4, f"Self-similarity should be ~1.0, got {scores[0][0]}"
    print(f"[OK] FAISS FlatIP cosine similarity index test passed! Top score: {scores[0][0]:.4f}")

if __name__ == "__main__":
    sample_pdf = os.path.join(current_dir, "sample_apollo.pdf")
    create_sample_pdf(sample_pdf)
    test_extraction_and_chunking(sample_pdf)
    test_faiss_indexing()
    print("\n[SUCCESS] All automated tests passed successfully!")
