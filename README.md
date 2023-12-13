# LLM Enhanced Knowledge Distillation Model

Our model is a state-of-the-art framework for cold-start recommendation, without assuming that warm and cold items are independent and identical distributed (IID). It replaces fixed neural network in student model by dynamic layers, whose parameters are decided by item's content features.

# Quick Start

Run `python main.py 'dataset' 'function'` replace 'dataset' and 'function' by your own choice.

1. Datasets: You can choose from "ADLI" or "Recommender"
   - CiteULike: assuming warm and cold items are iid
   - Amazon2018: without iid assumption
2. Functions:
   - train_teacher: Baysian Personalized Ranking Matrix Factorization (BPRMF)
   - test_teacher
   - train_student: MLP as student model
   - test_student
   - train_meta: dynamic layers
   - test_meta

# Postscript

`train_meta` and `test_meta` is currently running slower than expected.The issue is recognized, and I'm actively working on investigating the root causes. This involves profiling the function to understand the performance bottlenecks and exploring optimization strategies.
