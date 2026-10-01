"""
Personalized Learning Path Recommender for Interns
Model  : Collaborative Filtering (Matrix Factorization with biases, trained via SGD)
Extras : Course metadata (level, prerequisites, skills) used for filtering + cold start
Run    : python intern_learning_recommender.py
Deps   : pip install numpy pandas
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# 1. DATA  (apni real data se replace karein: do CSV files)
#    courses.csv      -> course_id, title, skill, level (1-3), prerequisite (course_id ya NaN)
#    interactions.csv -> intern_id, course_id, rating (1-5)   # rating = feedback ya completion score
# ----------------------------------------------------------------------------
def make_sample_data(n_interns=60):
    skills = ["Python", "SQL", "ML", "Cloud", "Communication", "Git"]
    rows, cid = [], 0
    for s in skills:
        prev = np.nan
        for lvl in (1, 2, 3):
            rows.append(dict(course_id=cid, title=f"{s} L{lvl}", skill=s, level=lvl, prerequisite=prev))
            prev = cid
            cid += 1
    courses = pd.DataFrame(rows)

    # Interns ke hidden interest groups (synthetic patterns)
    inter = []
    for i in range(n_interns):
        fav = rng.choice(skills, size=2, replace=False)
        done = rng.choice(courses.course_id, size=rng.integers(5, 10), replace=False)
        for c in done:
            sk = courses.loc[c, "skill"]
            base = 4.3 if sk in fav else 2.5
            inter.append((i, c, float(np.clip(round(base + rng.normal(0, 0.6)), 1, 5))))
    interactions = pd.DataFrame(inter, columns=["intern_id", "course_id", "rating"])
    return courses, interactions


# ----------------------------------------------------------------------------
# 2. MODEL: Matrix Factorization   r_ui ≈ mu + b_u + b_i + p_u · q_i
# ----------------------------------------------------------------------------
class MatrixFactorization:
    def __init__(self, n_factors=8, lr=0.01, reg=0.05, epochs=60):
        self.k, self.lr, self.reg, self.epochs = n_factors, lr, reg, epochs

    def fit(self, df, n_users, n_items):
        self.mu = df.rating.mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        data = df[["intern_id", "course_id", "rating"]].to_numpy()

        for ep in range(self.epochs):
            rng.shuffle(data)
            for u, i, r in data:
                u, i = int(u), int(i)
                err = r - self.predict(u, i)
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def predict_all(self, u):
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]


# ----------------------------------------------------------------------------
# 3. RECOMMENDER (model + metadata rules)
# ----------------------------------------------------------------------------
class LearningPathRecommender:
    def __init__(self, courses, interactions):
        self.courses = courses.set_index("course_id")
        self.inter = interactions
        self.n_users = interactions.intern_id.max() + 1
        self.n_items = len(courses)
        self.model = MatrixFactorization().fit(interactions, self.n_users, self.n_items)
        self.popularity = interactions.groupby("course_id").rating.mean()

    def _completed(self, intern_id):
        return set(self.inter.loc[self.inter.intern_id == intern_id, "course_id"])

    def recommend(self, intern_id, top_n=5):
        done = self._completed(intern_id)

        # Cold start: naya intern (koi history nahi) -> beginner + popular courses
        if intern_id >= self.n_users or not done:
            beg = self.courses[self.courses.level == 1].index
            ranked = self.popularity.reindex(beg).fillna(0).sort_values(ascending=False)
            return self._format(ranked.index[:top_n], ranked.values[:top_n])

        scores = self.model.predict_all(intern_id)
        out = []
        for cid in np.argsort(-scores):
            if cid in done:
                continue
            pre = self.courses.loc[cid, "prerequisite"]
            if pd.notna(pre) and int(pre) not in done:      # prerequisite complete nahi
                continue
            out.append((cid, scores[cid]))
            if len(out) == top_n:
                break
        return self._format([c for c, _ in out], [s for _, s in out])

    def _format(self, ids, scores):
        df = self.courses.loc[list(ids), ["title", "skill", "level"]].copy()
        df["predicted_score"] = np.round(np.clip(scores, 1, 5), 2)
        return df.reset_index()


# ----------------------------------------------------------------------------
# 4. EVALUATION (train/test split)
# ----------------------------------------------------------------------------
def evaluate(courses, interactions, k=5):
    test = interactions.sample(frac=0.2, random_state=1)
    train = interactions.drop(test.index)
    n_users, n_items = interactions.intern_id.max() + 1, len(courses)
    m = MatrixFactorization().fit(train, n_users, n_items)

    preds = np.array([m.predict(int(u), int(i)) for u, i in zip(test.intern_id, test.course_id)])
    rmse = np.sqrt(np.mean((np.clip(preds, 1, 5) - test.rating.values) ** 2))

    # Hit-rate@K: kya intern ka pasandida (rating>=4) test course top-K me aaya?
    hits, total = 0, 0
    for u, grp in test[test.rating >= 4].groupby("intern_id"):
        seen = set(train.loc[train.intern_id == u, "course_id"])
        s = m.predict_all(u)
        s[list(seen)] = -np.inf
        topk = set(np.argsort(-s)[:k])
        hits += len(topk & set(grp.course_id))
        total += len(grp)
    return rmse, hits / max(total, 1)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    courses, interactions = make_sample_data()
    # Real data ke liye:
    # courses = pd.read_csv("courses.csv"); interactions = pd.read_csv("interactions.csv")

    rmse, hr = evaluate(courses, interactions)
    print(f"RMSE: {rmse:.3f} | HitRate@5: {hr:.3f}\n")

    rec = LearningPathRecommender(courses, interactions)
    for intern in (0, 7):
        print(f"--- Recommended modules for intern {intern} ---")
        print(rec.recommend(intern, top_n=5).to_string(index=False), "\n")

    print("--- New intern (cold start) ---")
    print(rec.recommend(9999, top_n=3).to_string(index=False))
