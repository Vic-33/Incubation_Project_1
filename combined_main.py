import speech_recognition as sr
import pyttsx3
import csv
import ahocorasick
import pandas as pd
import numpy as np
import time
import os

# Clear or initialize the recognized_text.csv and order_history.csv files
def initialize_files():
    with open('recognized_text.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Recognized Text"])  # Initialize with header

    if not os.path.exists('order_history.csv'):
        with open('order_history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Order History"])  # Initialize with header if not present

initialize_files()

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Step 1: Voice Recognition and Saving to CSV
def recognize_and_save_to_csv():
    recognized_text = []
    try:
        with open('recognized_text.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            print("Listening for your order...")

            while True:
                try:
                    with sr.Microphone() as mic:
                        recognizer.adjust_for_ambient_noise(mic, duration=1.0)
                        print("Please say your order.")
                        audio = recognizer.listen(mic, timeout=5)

                        # Convert audio to text
                        text = recognizer.recognize_google(audio).lower()
                        recognized_text.append(text)
                        writer.writerow([text])
                        print(f"Recognized: {text}")

                except sr.UnknownValueError:
                    print("Could not understand audio. Retrying...")
                    continue
                except sr.WaitTimeoutError:
                    print("Listening timed out.")
                    continue
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

                # Allow additional input by breaking only after confirmation step
                time.sleep(2)
                break

    except FileNotFoundError as e:
        print("Error: recognized_text.csv file could not be found or created.")
    
    return recognized_text


# Function to ask if the user wants to order anything else
def ask_if_continue_ordering(final_order):
    text_to_speech(f"You ordered: {', '.join(final_order)}. Do you want to order anything else?")

    try:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1.0)
            audio = recognizer.listen(mic)
            response = recognizer.recognize_google(audio).lower()
            print(f"User response: {response}")

            if 'no' in response or 'stop' in response:
                return False
            else:
                return True

    except sr.UnknownValueError:
        print("Could not understand response.")
        return True  # Assume user wants to continue if response is unclear
    except Exception as e:
        print(f"An error occurred during confirmation: {e}")
        return False  # Assume exit on critical error


# Step 2: Convert CSV to Text File
def csv_to_text(input_csv, output_txt):
    try:
        with open(input_csv, 'r') as csv_file, open(output_txt, 'w') as text_file:
            for line in csv_file:
                text_file.write(line.replace(',', ' '))
    except FileNotFoundError:
        print("Error: CSV file not found when trying to convert to text.")


# Step 3: Build Aho-Corasick Automaton
def build_aho_corasick(keywords):
    automaton = ahocorasick.Automaton()
    for idx, keyword in enumerate(keywords):
        automaton.add_word(keyword, (idx, keyword))
    automaton.make_automaton()
    return automaton


# Step 4: Recognize Keywords in Text File
def recognize_keywords_in_text(file_path, keywords):
    automaton = build_aho_corasick(keywords)
    recognized_words = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                for end_index, (idx, original_value) in automaton.iter(line):
                    recognized_words.add(original_value)
    except FileNotFoundError:
        print("Error: Text file not found for keyword recognition.")
    
    return recognized_words


# Step 5: Text-to-Speech Confirmation
def text_to_speech(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {e}")


# Step 6: Update Order History for Recommendation Model
def update_order_history(order_items):
    try:
        with open('order_history.csv', 'a', newline='') as history_file:
            writer = csv.writer(history_file)
            writer.writerow(order_items)
    except Exception as e:
        print(f"Error updating order history: {e}")


# Step 7: Build Co-Occurrence Matrix for Recommendations
def build_cooccurrence_matrix(order_history, items):
    item_index = {item: idx for idx, item in enumerate(items)}
    co_matrix = np.zeros((len(items), len(items)), dtype=int)

    for order in order_history:
        indices = [item_index[item] for item in order if item in item_index]
        for i in indices:
            for j in indices:
                if i != j:
                    co_matrix[i][j] += 1

    return co_matrix, item_index


# Step 8: Generate Recommendations Based on Current Order
def recommend_items(current_order, co_matrix, item_index, items, top_n=3):
    recommendations = {}
    indices = [item_index[item] for item in current_order if item in item_index]

    for idx in indices:
        similar_items = co_matrix[idx]
        for item_idx, score in enumerate(similar_items):
            if score > 0 and items[item_idx] not in current_order:
                recommendations[items[item_idx]] = recommendations.get(items[item_idx], 0) + score

    recommended_items = sorted(recommendations, key=recommendations.get, reverse=True)[:top_n]
    return recommended_items


# Step 9: Main Routine
def main():
    if not os.path.exists('Menu.csv'):
        print("Error: Menu.csv file is required but not found.")
        return

    menu_df = pd.read_csv('Menu.csv')
    menu_items = menu_df['Item'].tolist()

    final_order = []

    while True:
        recognized_text = recognize_and_save_to_csv()
        final_order.extend(recognized_text)

        csv_to_text('recognized_text.csv', 'output.txt')

        recognized_items = recognize_keywords_in_text('output.txt', menu_items)
        print("Recognized Menu Items:", recognized_items)

        if recognized_items:
            confirmation_text = f"You ordered: {', '.join(recognized_items)}."
            text_to_speech(confirmation_text)
            update_order_history(recognized_items)

            if not ask_if_continue_ordering(final_order):
                break
        else:
            print("No items recognized. Please try again.")

    final_confirmation_text = f"Your final order is: {', '.join(final_order)}."
    text_to_speech(final_confirmation_text)

    try:
        order_history = pd.read_csv('order_history.csv').dropna().values.tolist()
    except FileNotFoundError:
        print("Error: Order history file is missing.")
        return

    co_matrix, item_index = build_cooccurrence_matrix(order_history, menu_items)
    recommendations = recommend_items(final_order, co_matrix, item_index, menu_items)
    print("Recommended Items:", recommendations)

    recommendation_text = f"We also recommend: {', '.join(recommendations)}."
    text_to_speech(recommendation_text)

if __name__ == "__main__":
    main()












