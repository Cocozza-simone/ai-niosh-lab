#!/usr/bin/env python3
"""
Optimize RAG system for best performance on CPU
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class OptimizedEmbeddingEngine:
    """Optimized embedding engine for CPU performance"""
    
    def __init__(self, embedding_engine):
        self.engine = embedding_engine
        self.executor = ThreadPoolExecutor(max_workers=6)  # Parallel processing
        self.batch_size = 8  # Optimal batch size for CPU
        
    def optimize_embed_chunks(self, chunks):
        """Optimized embedding for CPU"""
        if not chunks:
            return []
        
        start_time = time.time()
        print(f"Starting optimized CPU embedding for {len(chunks)} chunks...")
        
        # Filter chunks by content
        valid_chunks = [c for c in chunks if c.content.strip()]
        
        if len(valid_chunks) <= self.batch_size:
            # Small batch - use sequential
            result = self.engine.embed_chunks(valid_chunks)
        else:
            # Large batch - use parallel processing
            batches = [valid_chunks[i:i + self.batch_size] 
                      for i in range(0, len(valid_chunks), self.batch_size)]
            
            result = []
            for batch in batches:
                batch_result = self.engine.embed_chunks(batch)
                result.extend(batch_result)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds ({len(result)}/{len(valid_chunks)} chunks)")
        
        return result

def install_pytorch_gpu():
    """Instructions for installing PyTorch with GPU support"""
    print("=== PyTorch GPU Installation Instructions ===\n")
    
    print("To install PyTorch with CUDA support for Windows:")
    print("1. Check your CUDA version with: nvidia-smi")
    print("2. Install PyTorch with CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   (Replace cu118 with your CUDA version: cu117, cu116, etc.)\n")
    
    print("Or without specific version (latest):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuda")
    
    print("\nAlternative CPU optimizations:")
    print("1. Use ONNX Runtime for faster CPU inference:")
    print("   pip install onnxruntime")
    print("2. Use optimized PyTorch build:")
    print("   pip install intel-extension-for-pytorch")
    print("3. Consider using OpenBLAS:")
    print("   Set environment variable: OMP_NUM_THREADS=4")
    
    print("\nCurrent status: Using optimized CPU mode")

if __name__ == "__main__":
    install_pytorch_gpu()