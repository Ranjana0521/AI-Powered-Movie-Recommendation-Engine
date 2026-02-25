# 🎬 Hybrid Movie Recommendation System

An intelligent **Hybrid Movie Recommendation System** built using **Collaborative Filtering and Content-Based Filtering**, deployed with **Streamlit**.

This project demonstrates how modern recommendation engines (like Netflix and Amazon) personalize content for users using Machine Learning techniques.

---

# 📌 Project Overview

This system recommends movies by combining:

* 👥 Collaborative Filtering (User–User Similarity)
* 🎭 Content-Based Filtering (Genre Similarity using TF-IDF)
* 🔥 Hybrid Recommendation Strategy

By combining both approaches, the system improves recommendation accuracy, personalization, and diversity.

---

# 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TF-IDF Vectorization
* Cosine Similarity

---

# 📊 Dataset Used

Dataset: **MovieLens (ml-latest-small)**

Contains:

* 9,000+ Movies
* 100,000+ Ratings
* 600+ Users

Files used:

* `movies.csv`
* `ratings.csv`

---

# ⚙️ System Architecture

The project consists of three main components:

## 1️⃣ Data Preprocessing

* Loaded datasets using Pandas
* Merged movies and ratings data
* Created User–Movie rating matrix
* Handled missing values using zero imputation

---

## 2️⃣ Collaborative Filtering

Collaborative filtering recommends movies based on similar users.

Steps:

* Created User-Movie matrix
* Computed cosine similarity between users
* Found most similar users
* Recommended movies liked by similar users

Formula used:

Cosine Similarity = (A · B) / (||A|| × ||B||)

---

## 3️⃣ Content-Based Filtering

Content-based filtering recommends movies similar in genre.

Steps:

* Extracted movie genres
* Applied TF-IDF Vectorization
* Computed cosine similarity between movies
* Returned most similar movies

---

## 4️⃣ Hybrid Recommendation Strategy

To improve performance:

* Generated collaborative recommendations
* Generated content-based recommendations
* Combined both lists
* Removed duplicates
* Returned Top-N movies

This improves:

* Accuracy
* Personalization
* Recommendation diversity

---

# 🖥️ Streamlit Web Application

The project is deployed using Streamlit.

Features:

* User-friendly interface
* Movie selection dropdown
* Real-time recommendations
* Clean UI with styled movie cards
* Displays rating and year

Run the app:

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```
Hybrid-Movie-Recommendation-System/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│
├── model/
│   └── recommender.py
│
├── app.py
├── requirements.txt

```

---

# 🚀 Installation Guide

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Hybrid-Movie-Recommendation-System.git
cd Hybrid-Movie-Recommendation-System
```

## Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the Application

```bash
streamlit run app.py
```
---

# 🎯 Key Learnings

* Understanding recommendation system architectures
* Implementing hybrid ML models
* Using cosine similarity for similarity measurement
* Working with real-world datasets
* Building and deploying ML apps using Streamlit
