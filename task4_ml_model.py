# ============================================================
# CODTECH INTERNSHIP - TASK 4
# Machine Learning Model Implementation
# Author: [Your Name]
# Description: Spam Email Detection using multiple ML models.
#              Includes EDA, preprocessing, model comparison,
#              hyperparameter tuning, and evaluation metrics.
# ============================================================

# ── IMPORTS ──
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib

print("=" * 60)
print("  CODTECH INTERNSHIP — TASK 4")
print("  Spam Email Detection using Machine Learning")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1: GENERATE REALISTIC DATASET
# ─────────────────────────────────────────
print("\n[1/6] Generating dataset...")

np.random.seed(42)

spam_messages = [
    "Congratulations! You have won a $1000 gift card. Click here to claim now!",
    "FREE iPhone 15! Limited offer. Act now before it expires!",
    "URGENT: Your bank account has been compromised. Verify immediately.",
    "You are selected for a cash prize of $5000. Send your details now.",
    "Make money fast! Work from home and earn $500 per day easily.",
    "Dear winner, you have been selected for our lottery prize claim today.",
    "Get cheap medications online. No prescription needed. Order now!",
    "Hot singles in your area want to meet you tonight. Click here.",
    "Lose 30 pounds in 30 days! Miracle weight loss pill now available.",
    "Your account is suspended. Update your information immediately.",
    "Buy cheap Rolex watches! 90% off original price. Limited stock!",
    "Earn money online fast! No experience needed. Work from anywhere.",
    "You've been pre-approved for a loan of up to $50,000. Apply now!",
    "Exclusive investment opportunity! Double your money in 30 days.",
    "Click here to unsubscribe from future messages. Act fast!",
    "Nigerian prince needs help transferring $10 million. Huge reward.",
    "Your PayPal account will be closed unless you verify by clicking here.",
    "Get pills online without prescription. Best prices guaranteed!",
    "Final warning: Pay your overdue bill or face legal action immediately.",
    "Congratulations! You are our lucky customer of the month. Claim prize.",
    "FREE trial! No credit card required. Sign up and start earning today.",
    "Your mortgage has been approved. Refinance now and save thousands.",
    "Sex dating site - find local singles tonight. No strings attached.",
    "Buy followers, likes, views - grow your social media fast. Cheap!",
    "Special alert! Your system is infected. Download antivirus now free.",
] * 8  # Repeat to get more data

ham_messages = [
    "Hey, are we still on for the team meeting tomorrow at 10 AM?",
    "I've reviewed your project proposal and have a few suggestions.",
    "Please find the attached invoice for the services rendered last month.",
    "Can you send me the report by Friday? The client is waiting.",
    "Happy birthday! Hope you have a wonderful day with your family.",
    "The conference call has been rescheduled to 3 PM on Thursday.",
    "I'll be working from home today due to a family emergency.",
    "Thanks for your help with the project. Really appreciated your effort.",
    "Can we reschedule our lunch meeting to next week? I'm quite busy.",
    "Please review the attached document and provide your feedback.",
    "The software update has been deployed successfully to production.",
    "I need your approval for the budget request before end of day.",
    "Looking forward to meeting you at the conference next month.",
    "Your subscription has been renewed successfully. Thank you!",
    "The quarterly report is ready. Please find it attached to this email.",
    "Reminder: Annual performance review meeting on 25th at 2 PM.",
    "Your order #12345 has been shipped and will arrive in 3 days.",
    "We are pleased to inform you that your application has been accepted.",
    "Please confirm your attendance for the department meeting tomorrow.",
    "The new library resources are now available on the student portal.",
    "Your leave request for December 24-26 has been approved.",
    "Please update your emergency contact information in the HR system.",
    "The project deadline has been extended by one week as discussed.",
    "Thank you for attending the workshop. Here are the slides shared.",
    "Your assignment has been graded. Please check the student portal.",
] * 8

# Combine and shuffle
all_messages = spam_messages + ham_messages
all_labels = ["spam"] * len(spam_messages) + ["ham"] * len(ham_messages)
combined = list(zip(all_messages, all_labels))
np.random.shuffle(combined)
messages, labels = zip(*combined)

df = pd.DataFrame({"message": messages, "label": labels})
df["length"] = df["message"].apply(len)
df["word_count"] = df["message"].apply(lambda x: len(x.split()))
df.to_csv("task4_email_dataset.csv", index=False)
print(f"  ✓ Dataset: {len(df)} emails ({df['label'].value_counts()['spam']} spam, {df['label'].value_counts()['ham']} ham)")

# ─────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────
print("\n[2/6] Generating EDA visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Spam Email Detection — Exploratory Data Analysis", fontsize=15, fontweight="bold")

# Plot 1: Class distribution
label_counts = df["label"].value_counts()
colors = ["#e74c3c", "#2ecc71"]
axes[0, 0].pie(label_counts.values, labels=label_counts.index,
               autopct="%1.1f%%", colors=colors, startangle=90,
               textprops={"fontsize": 12})
axes[0, 0].set_title("Email Class Distribution", fontweight="bold")

# Plot 2: Message length distribution
for label, color in zip(["spam", "ham"], ["#e74c3c", "#2ecc71"]):
    subset = df[df["label"] == label]["length"]
    axes[0, 1].hist(subset, bins=30, alpha=0.6, color=color, label=label.upper(), edgecolor="white")
axes[0, 1].set_title("Message Length Distribution", fontweight="bold")
axes[0, 1].set_xlabel("Character Count")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()

# Plot 3: Word count comparison
df.groupby("label")["word_count"].plot(kind="kde", ax=axes[1, 0],
                                        color=["#e74c3c", "#2ecc71"])
axes[1, 0].set_title("Word Count Density by Class", fontweight="bold")
axes[1, 0].set_xlabel("Word Count")
axes[1, 0].legend(["Spam", "Ham"])

# Plot 4: Average length by class
avg_len = df.groupby("label")[["length", "word_count"]].mean()
avg_len.plot(kind="bar", ax=axes[1, 1], color=["#3498db", "#e67e22"],
             edgecolor="white", rot=0)
axes[1, 1].set_title("Average Length & Word Count by Class", fontweight="bold")
axes[1, 1].set_ylabel("Average Count")
axes[1, 1].legend(["Avg Length (chars)", "Avg Word Count"])

plt.tight_layout()
plt.savefig("task4_eda.png", dpi=130, bbox_inches="tight")
print("  ✓ EDA saved as 'task4_eda.png'")

# ─────────────────────────────────────────
# STEP 3: PREPROCESSING & TRAIN/TEST SPLIT
# ─────────────────────────────────────────
print("\n[3/6] Preprocessing data...")

le = LabelEncoder()
y = le.fit_transform(df["label"])   # spam=1, ham=0
X = df["message"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────
# STEP 4: TRAIN & COMPARE 5 MODELS
# ─────────────────────────────────────────
print("\n[4/6] Training and comparing 5 models...")

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", MultinomialNB())
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", LinearSVC(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
}

results = {}
print(f"\n  {'Model':<22} {'Accuracy':>10} {'CV Mean':>10} {'CV Std':>8}")
print("  " + "-"*54)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred,
        "model": model
    }
    print(f"  {name:<22} {acc:>10.4f} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f}")

# ─────────────────────────────────────────
# STEP 5: HYPERPARAMETER TUNING (Best Model)
# ─────────────────────────────────────────
print("\n[5/6] Hyperparameter tuning for Logistic Regression...")

param_grid = {
    "tfidf__max_features": [3000, 5000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}
grid_search = GridSearchCV(models["Logistic Regression"], param_grid,
                           cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_acc = accuracy_score(y_test, best_model.predict(X_test))
print(f"  ✓ Best params: {grid_search.best_params_}")
print(f"  ✓ Tuned accuracy: {best_acc:.4f}")

# Save the best model
joblib.dump(best_model, "task4_spam_model.pkl")
print("  ✓ Best model saved as 'task4_spam_model.pkl'")

# ─────────────────────────────────────────
# STEP 6: EVALUATION VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[6/6] Generating evaluation charts...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle("ML Model Evaluation — Spam Detection", fontsize=15, fontweight="bold")

# Chart 1: Model accuracy comparison
names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]
cv_means = [results[n]["cv_mean"] for n in names]
x = np.arange(len(names))
w = 0.35
bars1 = axes2[0, 0].bar(x - w/2, accuracies, w, label="Test Accuracy", color="#3498db", edgecolor="white")
bars2 = axes2[0, 0].bar(x + w/2, cv_means, w, label="CV Mean Accuracy", color="#e67e22", edgecolor="white")
axes2[0, 0].set_title("Model Accuracy Comparison", fontweight="bold")
axes2[0, 0].set_xticks(x)
axes2[0, 0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
axes2[0, 0].set_ylim(0.85, 1.01)
axes2[0, 0].set_ylabel("Accuracy")
axes2[0, 0].legend()
for bar, val in zip(bars1, accuracies):
    axes2[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.3f}", ha="center", fontsize=7)

# Chart 2: Confusion Matrix (best model)
best_name = max(results, key=lambda n: results[n]["accuracy"])
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=axes2[0, 1], colorbar=False, cmap="Blues")
axes2[0, 1].set_title(f"Confusion Matrix — {best_name}", fontweight="bold")

# Chart 3: ROC curves
for name, data in results.items():
    try:
        model = data["model"]
        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes2[1, 0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.5)
    except Exception:
        pass
axes2[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes2[1, 0].set_title("ROC Curves — All Models", fontweight="bold")
axes2[1, 0].set_xlabel("False Positive Rate")
axes2[1, 0].set_ylabel("True Positive Rate")
axes2[1, 0].legend(fontsize=7)

# Chart 4: Cross-validation scores per model
cv_data = {n: results[n]["cv_mean"] for n in names}
err_data = {n: results[n]["cv_std"] for n in names}
axes2[1, 1].barh(list(cv_data.keys()), list(cv_data.values()),
                  xerr=list(err_data.values()), color="#2ecc71",
                  edgecolor="white", capsize=4)
axes2[1, 1].set_title("Cross-Validation Scores (5-Fold)", fontweight="bold")
axes2[1, 1].set_xlabel("Mean Accuracy")
axes2[1, 1].set_xlim(0.85, 1.0)

plt.tight_layout()
plt.savefig("task4_evaluation.png", dpi=130, bbox_inches="tight")
print("  ✓ Evaluation saved as 'task4_evaluation.png'")

# ─────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  BEST MODEL: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
print(f"{'='*60}")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=le.classes_))

# ─────────────────────────────────────────
# LIVE PREDICTION DEMO
# ─────────────────────────────────────────
print("\n📬 Live Prediction Demo:")
test_emails = [
    "Congratulations! You won $10,000! Claim your prize now!",
    "Hi team, please review the attached project report before Friday.",
    "FREE gift card! Click here to claim your reward immediately!",
    "Meeting at 3 PM tomorrow in Conference Room B. Please confirm.",
]
for email in test_emails:
    pred = best_model.predict([email])[0]
    label = le.inverse_transform([pred])[0].upper()
    icon = "🚨" if label == "SPAM" else "✅"
    print(f"  {icon} [{label}] {email[:65]}...")

print("\n🎉 Task 4 Complete! All outputs saved.")
