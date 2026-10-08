import argparse
from pathlib import Path
import joblib
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--pclass", type=int, required=True)
parser.add_argument("--sex", required=True)
parser.add_argument("--age", type=float, required=True)
parser.add_argument("--fare", type=float, required=True)
parser.add_argument("--sibsp", type=int, default=0)
parser.add_argument("--parch", type=int, default=0)
parser.add_argument("--embarked", default="S")
args = parser.parse_args()

model = joblib.load(Path(__file__).resolve().parent / "models" / "titanic_survival.joblib")

row = pd.DataFrame([{
    "Pclass": args.pclass, "Sex": args.sex, "Age": args.age,
    "SibSp": args.sibsp, "Parch": args.parch,
    "Fare": args.fare, "Embarked": args.embarked,
}])

prob = model.predict_proba(row)[0, 1]
print(f"Survival probability: {prob:.3f}")
print("Prediction:", "survived" if prob >= 0.5 else "did not survive")
