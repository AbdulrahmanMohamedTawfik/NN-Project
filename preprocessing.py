import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# uncomment this in first run
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_arabic_text(text):
    # Remove non-Arabic characters and numbers
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    # Remove repetitive characters like حككككككككزنمظخؤحسدسسدسظسز
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Remove specific patterns like "٠٠٠" and empty strings
    text = re.sub(r'٠+', '', text)
    text = re.sub(r'""', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('arabic'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


file_path = 'train.csv'

df = pd.read_csv(file_path)

df['preprocessed_review'] = df['review_description'].apply(preprocess_arabic_text)

df = df.dropna(subset=['preprocessed_review'])

print("Original Data:")
print(df[['review_description']].head())

print("\nPreprocessed Data:")
print(df[['preprocessed_review']].head())

output_file_path = 'preprocessed_file.txt'

# Save to text
df[['preprocessed_review']].to_csv(output_file_path, index=False, sep='\t', encoding='utf-8')

print(f"\nPreprocessed data saved to: {output_file_path}")
