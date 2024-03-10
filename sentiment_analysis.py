import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import random

# Loading spaCy model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Reading the csv file
reviews_df = pd.read_csv("amazon_product_reviews.csv")
# Extracting information about the file
reviews_df.info()

# Preprocessing the dataframe
# Dropping the rows with missing review text
dropped_na_df = reviews_df.dropna(subset=['reviews.text'])
# Saving the relevant reviews text into a new dataframe of reviews data only
reviews_data_df = dropped_na_df['reviews.text']


# Cleaning and preprocessing the review text
def clean_review(product_review):
    # Processing the product review text through spaCy
    doc = nlp(product_review)
    # Creating a list of lemmatized tokens, where the tokens are not stop words
    # or punctuation. The lemmatized tokens are converted to lowercase to
    # ensure consistency.
    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop 
                      and not token.is_punct]
    # Joining the cleaned tokens back into a string
    cleaned_review = ' '.join(cleaned_tokens)
    return cleaned_review


# Function performing sentiment analysis of a product review
def sentiment_analysis (product_review):
    doc = nlp(product_review)

    # Noting that in the context of SpacyTextBlob sentiment refers to a
    # a combination of polarity and subjectivity, the two will be stored
    # separately in two variables to facilitate further data analysis.

    # review_polarity measures the score of how positive/negative the product
    # review is, within the range [-1.0, 1.0]
    review_polarity = doc._.blob.sentiment.polarity

    # review_subjectivity measures the presence of subjective content within
    # the product review, within the range [0.0, 1.0]
    review_subjectivity = doc._.blob.sentiment.subjectivity
    # The function returns a tuple of the polarity and subjectivity scores
    return review_polarity, review_subjectivity


def main():

    # Applying the clean_review function to the reviews dataframe
    clean_reviews = reviews_data_df.apply(clean_review)

    # Applying the sentiment_analysis function to the clean reviews
    reviews_sentiment = clean_reviews.apply(sentiment_analysis)

    # Creating a dataframe that stores the clean review text and the 
    # associated sentiment (polarity and subjectivity)
    sentiment_df = pd.DataFrame({
        'Review': clean_reviews,
        'Polarity': reviews_sentiment.apply(lambda x: x[0]),
        'Subjectivity': reviews_sentiment.apply(lambda x: x[1])
    })

    print(sentiment_df.head())

    # Calculating averages for polarity and subjectivity and printing the 
    # rounded results for better readability
    average_polarity = round(sentiment_df['Polarity'].mean(), 2)
    average_subjectivity = round(sentiment_df['Subjectivity'].mean(), 2)
    print(f"The average polarity score of the reviews is: "
          f"{average_polarity}")
    print(f"The average subjectivity score of the reviews is: "
          f"{average_subjectivity}")

    # Counting positive and negative reviews and printing the results
    negative_reviews = sentiment_df[sentiment_df['Polarity'] < 0]
    num_negative_reviews = negative_reviews.shape[0]

    positive_reviews = sentiment_df[sentiment_df['Polarity'] > 0]
    num_positive_reviews = positive_reviews.shape[0]

    print(f"The total number of positive reviews is: {num_positive_reviews}")
    print(f"The total number of negative reviews is: {num_negative_reviews}")


    # Calculating the correlation between polarity and subjectivity. This is 
    # used to understand if the two variables change together. The coefficient 
    # can have values in the range [-1.0, 1,0] where -1 = negative correlation,
    # 0 = no correlation, 1 = positive correlation
    correlation = sentiment_df['Polarity'].corr(sentiment_df['Subjectivity'])
    print(f"The correlation score between polarity and subjectivity is: "
          f"{round(correlation, 2)}")


    # Calculating the average subjectivity in negative/positive reviews
    average_subjectivity_negative = negative_reviews['Subjectivity'].mean()
    average_subjectivity_positive = positive_reviews['Subjectivity'].mean()
    print(f"The average subjectivity in negative reviews: "
          f"{round(average_subjectivity_negative, 2)}")
    print(f"The average subjectivity in positive reviews: "
          f"{round(average_subjectivity_positive, 2)}")


    # Similarity
    # Calculating the similarity between all reviews is very computationally
    # intensive and therefore, for this exercise, only small samples of the 
    # dataset are used.
    
    # Randomly select two samples of positive and negative reviews, using
    # pandas' sample method
    pos_reviews_sample = positive_reviews['Review'].sample(2)
    neg_reviews_sample = negative_reviews['Review'].sample(2)

    # Converting the sample reviews into a spaCy document
    doc_pos_1 = nlp(pos_reviews_sample.iloc[0])
    doc_pos_2 = nlp(pos_reviews_sample.iloc[1])
    doc_neg_1 = nlp(neg_reviews_sample.iloc[0])
    doc_neg_2 = nlp(neg_reviews_sample.iloc[1])

    # Calculating and printing similarity between the positive and between the 
    # negative reviews
    similarity_pos = doc_pos_1.similarity(doc_pos_2)
    similarity_neg = doc_neg_1.similarity(doc_neg_2)
    print(f"The similarity between two random positive reviews is: " 
          f"{round(similarity_pos, 2)}")
    print(f"The similarity between two random negative reviews is: "
          f"{round(similarity_neg, 2)}")

    # Calculating and printing similarity between the selected positive/
    # negative review
    similarity_pos_neg_1 = doc_pos_1.similarity(doc_neg_1)
    similarity_pos_neg_2 = doc_pos_2.similarity(doc_neg_2)
    print(f"The similarity between the first random positive and the first "
          f"random negative review is: {round(similarity_pos_neg_1, 2)}")
    print(f"The similarity between the second random positive and the second "
          f"random negative review is: {round(similarity_pos_neg_2, 2)}")


main()
    
