# anchored_akinator.py
import pandas as pd
import time

DATASET_FILE = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\akiantor_generated_dataset.csv"
QUESTIONS_FILE = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\akiantor_questions.csv"

MAX_QUESTIONS = 10

def load_data():
    dataset = pd.read_csv(DATASET_FILE)
    questions = pd.read_csv(QUESTIONS_FILE)

    # rename first column as "condition"
    condition_col = dataset.columns[0]
    dataset = dataset.rename(columns={condition_col: "condition"})

    # ensure binary encoding
    for col in dataset.columns[1:]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(0).astype(int)

    if not {"symptom", "question"}.issubset(questions.columns):
        raise ValueError("❌ 'akiantor_questions.csv' must have 'symptom' and 'question' columns")

    return dataset, questions

class AnchoredAkinator:
    def __init__(self, dataset, questions, main_problem):
        self.dataset = dataset
        self.questions = questions
        self.asked = set()
        self.num_questions = 0
        self.last_asked = None

        # anchor: filter only conditions where main_problem symptom = 1
        symptom_match = None
        for col in dataset.columns[1:]:
            if main_problem.lower() in col.lower():
                symptom_match = col
                break

        if symptom_match:
            self.remaining = dataset[dataset[symptom_match] == 1].copy()
        else:
            self.remaining = dataset.copy()

    def get_next_symptom(self):
        if len(self.remaining["condition"].unique()) <= 1 or self.num_questions >= MAX_QUESTIONS:
            return None

        # candidate symptoms not yet asked
        candidates = [c for c in self.remaining.columns[1:] if c not in self.asked]

        # score = how much this splits the remaining conditions
        scores = {}
        for col in candidates:
            counts = self.remaining.groupby("condition")[col].sum()
            if counts.nunique() > 1:
                scores[col] = counts.var()
        if not scores:
            return None
        return max(scores, key=scores.get)

    def ask_question(self, symptom):
        # if no symptom provided, do not produce a question
        if symptom is None:
            self.last_asked = None
            return None

        # remember which symptom we just asked so updates map to it
        self.last_asked = symptom
        row = self.questions[self.questions["symptom"] == symptom]
        if not row.empty:
            q = str(row["question"].values[0])
            # sanitize common HTML entities and tags that might come from CSV
            q = q.replace('<br>', ' ').replace('<br/>', ' ').replace('\r', ' ').replace('\n', ' ')
            q = ' '.join(q.split())
            return q
        # fallback readable question
        return f"Do you have {symptom.replace('_',' ')}?"

    def update(self, symptom, answer):
        # allow callers to pass None and fallback to the last asked symptom
        if symptom is None:
            symptom = self.last_asked
        if symptom is None:
            return

        self.asked.add(symptom)
        self.num_questions += 1

        a = (answer or '').strip().lower()
        if a == "yes":
            self.remaining = self.remaining[self.remaining[symptom] == 1]
        elif a == "no":
            self.remaining = self.remaining[self.remaining[symptom] == 0]
        else:
            # unexpected answer - don't filter but still record the symptom as asked
            pass

    def predict(self):
        conditions = self.remaining["condition"].unique()
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return f"Possibly one of: {', '.join(conditions[:3])}..."
        else:
            return None

    def precautions(self, condition):
        return [
            "Stay hydrated",
            "Take adequate rest",
            "Monitor temperature regularly",
            "Consult a doctor if symptoms worsen",
        ]

def main():
    dataset, questions = load_data()

    print("\nMedical Akinator Chatbot")
    print("I’ll ask yes/no questions to narrow down your condition.")
    print("Reply with: yes / no / nothing else. Type 'exit' anytime.\n")

    main_problem = input("Chatbot: Please describe your main problem.\nYou: ")
    aki = AnchoredAkinator(dataset, questions, main_problem)
    print("Chatbot: Got it. Let's explore further...")

    while True:
        if aki.num_questions >= MAX_QUESTIONS or len(aki.remaining["condition"].unique()) <= 1:
            break

        symptom = aki.get_next_symptom()
        if not symptom:
            break

        q = aki.ask_question(symptom)
        ans = input(f"Q{aki.num_questions+1}: {q}\nYou: ").strip().lower()

        if ans == "exit":
            print("Chatbot: Goodbye 👋")
            return
        if ans == "nothing else":
            break

        aki.update(symptom, ans)

    condition = aki.predict()
    if condition:
        print(f"\nLikely condition: {condition}")
        print("Suggested precautions:")
        for p in aki.precautions(condition):
            print("-", p)
    else:
        print("Could not confidently identify your condition. Please consult a doctor.")

if __name__ == "__main__":
    main()