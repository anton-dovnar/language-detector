import logging
import os
import requests
import pycountry
import regex
import collections
import pprint

from constants import LANGUAGE_CODES, URLS

logging.basicConfig(level=logging.INFO)


def download_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content.decode('utf-8', errors='ignore').strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading book from {url}: {e}")
        return None


def save_entity(entity, output_directory, language):
    os.makedirs(output_directory, exist_ok=True)
    filename = os.path.join(output_directory, f"{language}.txt")
    with open(filename, 'w', encoding='utf-8') as file:
        if isinstance(entity[0], list):
            for item in entity[0]:
                file.write(item)
        else:
            file.write(entity[0])


def split_and_pad(text):
    pattern = regex.compile(r"\b\p{L}+\'*\p{L}*\b", flags=regex.UNICODE)
    tokens = pattern.findall(text)
    padded_tokens = [token.lower() + '\n' for token in tokens]
    return padded_tokens


def data_cleaning(raw_text):
    regex_pattern = r"[，!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~\\0-9]"
    return regex.sub(regex_pattern, "", raw_text)


def skip_template_text(text):
    start_phrase = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_phrase_1 = "*** END OF THE PROJECT GUTENBERG EBOOK"
    end_phrase_2 = "End of Project Gutenberg"
    
    # Compile the regex pattern
    pattern = regex.compile(
        f'{regex.escape(start_phrase)}(.*?)(?:{regex.escape(end_phrase_1)}|{regex.escape(end_phrase_2)})', 
        regex.DOTALL
    )
    
    # Search for the pattern in the text
    is_match = pattern.search(text)
    
    if is_match:
        # Extract the relevant section of the text
        start_idx = is_match.start() + len(start_phrase)
        end_idx = is_match.end()
        intermediate_stage = text[start_idx:end_idx]
        
        # Find the next "***" and adjust the text
        next_start_idx = intermediate_stage.find("***")
        if next_start_idx != -1:
            text = intermediate_stage[next_start_idx + len("***") + 600:].strip()
    
    return data_cleaning(text)


def get_books_text(output_directory='', language_code="eng"):
    logging.info("Extracting Book Data")
    
    for index, url_list in enumerate(URLS):
        language = pycountry.languages.get(alpha_3=LANGUAGE_CODES[index]).name
        logging.info(f"Processing language: {language}")
        books_text = []
        cleaned_text = ""
        tokens = []

        for url in url_list:
            raw_text = download_text_from_url(url)
            if raw_text:
                cleaned_text += skip_template_text(raw_text)
                tokens.append(split_and_pad(cleaned_text))
                books_text.append(raw_text)

        save_entity(books_text, os.path.join(output_directory, "dataset"), language)
        save_entity(tokens, os.path.join(output_directory, "tokenized"), f"{language}.int1")

    logging.info("Book Data Extraction Completed")


def generate_ngrams(line):
    n = len(line)
    return [
        line[i:i+j] 
        for i in range(n) 
        for j in range(1, min(n - i + 1, 6))
    ]


def count_ngram_frequency(ngrams):
    return collections.Counter(ngrams)


def sort_ngrams_by_frequency(ngram_counter):
    return sorted(ngram_counter.items(), key=lambda x: (-x[1], x[0]))


def generate_and_count_ngrams(input_file_path, output_file_path, frequency_file_path):
    n_gram_counts = collections.Counter()

    # Read input file and count n-grams
    with open(input_file_path, 'r', encoding='UTF-8') as input_file:
        for line in input_file:
            line = line.strip()
            ngrams = generate_ngrams(line)
            n_gram_counts.update(ngrams)

    # Sort n-grams by frequency
    sorted_ngrams = sort_ngrams_by_frequency(n_gram_counts)

    # Write n-grams and their frequencies to output files
    with open(output_file_path, 'w', encoding='UTF-8') as output_file, \
         open(frequency_file_path, 'w', encoding='UTF-8') as frequency_file:

        for n_gram, count in sorted_ngrams:
            output_file.write(f"{n_gram}\n")
            frequency_file.write(f"{n_gram}: {count}\n")


def process_languages(language_codes):
    base_path = ""

    for code in language_codes:
        try:
            language = pycountry.languages.get(alpha_3=code).name
            input_file_path = os.path.join(base_path, "tokenized", f"{language}.int1.txt")
            n_gram_file_path = os.path.join(base_path, "nGrams", f"{language}.nGrams.txt")
            frequency_file_path = os.path.join(base_path, "processed", f"{language}.nGramsFrequency.txt")
            
            logging.info(f"Processing {language} for nGrams!")
            generate_and_count_ngrams(input_file_path, n_gram_file_path, frequency_file_path)
        
        except Exception as e:
            logging.error(f"Error processing {code} ({language}): {e}")


def preprocess_file(file_name):
    tokenized_lines = []
    with open(file_name, 'r', encoding='UTF-8') as file:
        for line in file:
            line = line.strip().replace(' ', '_ _')
            tokenized_lines.append(f'_{line}_\n')
    return tokenized_lines


def test_language(file_name):
    directory = "nGrams/"
    trained_language_files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.nGrams.txt')
    ]

    # Print out the list of trained languages
    logging.info("Trained language files:\n%s", pprint.pformat(trained_language_files))

    # Preprocess the input file and compute n-gram frequencies
    tokenized_lines = preprocess_file(file_name)
    n_gram_counts = collections.Counter()
    
    for line in tokenized_lines:
        ngrams = generate_ngrams(line)
        n_gram_counts.update(count_ngram_frequency(ngrams))

    sorted_ngrams = sort_ngrams_by_frequency(n_gram_counts)
    out_of_order_scores = []

    for language_file in trained_language_files:
        with open(language_file, 'r', encoding='UTF-8') as file:
            train_ngrams = [line.strip() for line in file]
        
        rank_table = {ngram: rank for rank, ngram in enumerate(train_ngrams, start=1)}
        score = sum(
            abs(rank_table.get(ngram, 50000) - rank)
            for rank, (ngram, _) in enumerate(sorted_ngrams, start=1)
        )
        out_of_order_scores.append(score)

    # Identify the language with the minimum score
    min_index = out_of_order_scores.index(min(out_of_order_scores))
    identified_language_file = trained_language_files[min_index]
    result = os.path.basename(identified_language_file).replace('.nGrams.txt', '')
    
    return result


if __name__ == "__main__":
    get_books_text()
    process_languages(LANGUAGE_CODES)

    file_name = "test.txt"
    lang_result = test_language(file_name)
    logging.info(f"Identified Language = {lang_result}")
