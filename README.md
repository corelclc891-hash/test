 Personalized Learning Path Recommender for Interns

A recommendation system that suggests custom learning modules to each intern based on their past learning patterns and course metadata.

It uses **Collaborative Filtering (Matrix Factorization)** and adds metadata rules (levels, prerequisites) to make the suggestions practical.

---

## 📌 Features

- **Matrix Factorization with biases**: `rating ≈ μ + b_user + b_item + P_u · Q_i`, trained with SGD
- **Prerequisite-aware**: a course is recommended only after its prerequisite is completed
- **No repeats**: already completed courses are filtered out
- **Cold-start handling**: new interns (no history) get popular beginner-level courses
- **Evaluation**: RMSE and Hit Rate@K on a train/test split
- **Lightweight**: only NumPy and Pandas, no heavy ML library needed

---

## 🗂 Project Structure

```
.
├── intern_learning_recommender.py   # Complete code (data, model, recommender, evaluation)
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install numpy pandas
```

---

## ▶️ Usage

```bash
python intern_learning_recommender.py
```

This will:
1. Generate sample data (or load your own)
2. Train the model and print RMSE and Hit Rate@5
3. Print top-5 recommended modules for sample interns
4. Show cold-start recommendations for a new intern

### Example Output

```
RMSE: 1.302 | HitRate@5: 0.231

--- Recommended modules for intern 0 ---
 course_id     title    skill  level  predicted_score
         0 Python L1   Python      1             3.52
        12 Communication L1 ...
```

> Note: the sample data is small and random, so accuracy is low. Real data will give better results.

---

## 📥 Using Your Own Data

In the `__main__` block, replace the sample data with your CSV files:

```python
courses = pd.read_csv("courses.csv")
interactions = pd.read_csv("interactions.csv")
```

**courses.csv** (course content metadata)

| course_id | title     | skill  | level | prerequisite |
|-----------|-----------|--------|-------|--------------|
| 0         | Python L1 | Python | 1     |              |
| 1         | Python L2 | Python | 2     | 0            |

**interactions.csv** (past intern learning patterns)

| intern_id | course_id | rating |
|-----------|-----------|--------|
| 0         | 0         | 5      |
| 0         | 3         | 4      |

`rating` (1-5) can be intern feedback or a score derived from completion/quiz results.
`intern_id` and `course_id` should be integers starting from 0.

---

## 🔧 Hyperparameters

Tune these in the `MatrixFactorization` class:

| Parameter   | Default | Meaning                              |
|-------------|---------|--------------------------------------|
| `n_factors` | 8       | Number of latent factors             |
| `lr`        | 0.01    | Learning rate                        |
| `reg`       | 0.05    | Regularization strength              |
| `epochs`    | 60      | Training passes over the data        |

---

## 🧠 How It Works

1. **Training**: the model learns a hidden "taste vector" for every intern and every course from past ratings.
2. **Scoring**: for an intern, it predicts a score for every course they haven't taken.
3. **Filtering**: courses whose prerequisites are not completed are removed.
4. **Ranking**: the top-N remaining courses become the personalized learning path.

---

## 🚀 Future Improvements

- Add a Streamlit dashboard or Flask/FastAPI endpoint
- Hybrid model using skill and text embeddings of course descriptions
- Implicit feedback (time spent, completion rate)
- Ordered multi-step learning paths instead of a flat top-N list

---

## 🛠 Tech Stack

Python · NumPy · Pandas

---

## 📄 License

MIT License. Free to use and modify.
