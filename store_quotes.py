from pymongo import MongoClient
import random

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Senior_project_db"]
quotes_collection = db["quotes"]

# Clear old quotes (optional)
quotes_collection.delete_many({})

# List of motivational quotes
quotes = [
    "Believe you can and you're halfway there.",
    "Push yourself, because no one else is going to do it for you.",
    "Don’t watch the clock; do what it does. Keep going.",
    "Success is what comes after you stop making excuses.",
    "The harder you work for something, the greater you’ll feel when you achieve it.",
    "Great things never come from comfort zones.",
    "Dream it. Wish it. Do it.",
    "Do something today that your future self will thank you for.",
    "The harder the struggle, the more glorious the triumph.",
    "It always seems impossible until it’s done.",
    "Your life does not get better by chance, it gets better by change.",
    "The expert in anything was once a beginner.",
    "Don’t watch the clock; do what it does. Keep going.",
    "Opportunities don’t happen. You create them.",
    "The biggest risk is not taking any risk.",
    "You miss 100% of the shots you don’t take.",
    "The best way to predict the future is to create it.",
    "Dream big and dare to fail."
]

# Insert quotes into MongoDB
for quote in quotes:
    quotes_collection.insert_one({"text": quote})

print("Quotes inserted successfully!")
