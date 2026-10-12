import json

notes = []
topics = ["AI study", "Workout", "Grocery", "Recipes", "Travel", "Books"]

for i in range(30):
    topic = topics[i % len(topics)]
    content = f"This is a note about {topic} number {i+1}. It contains important tips and details."
    notes.append({
        "title": f"{topic} note {i+1}",
        "content": content,
        "tags": [topic]
    })

with open("notes_dataset.json", "w") as f:
    json.dump(notes, f, indent=2)

print("✅ Dataset created: notes_dataset.json")
