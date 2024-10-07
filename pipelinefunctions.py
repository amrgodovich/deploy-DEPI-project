import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud

model = load_model('Amazon_review_sent_analysis model.h5')
model_ber=SentenceTransformer('all-MiniLM-L6-v2')

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
nltk.download('punkt')
stopwords_En = nltk.corpus.stopwords.words('english')
stopwords_En.remove('no')
stopwords_En.remove('not')

def preprocess(df):
    df['cleaned_Text'] = df['text'].apply(lambda x: clean_text(x))
    print("Done Cleaning")
    # df['embedding'] = df['cleaned_Text'].apply(lambda x: model_ber.encode(x))
    df['embedding'] = model_ber.encode(df['cleaned_Text'].tolist()).tolist()
    print("Done Embedding")
    return df

def prediction(df):
    # df['prediction'], df['probability'] = df['embedding'].apply(lambda x: predict_review_label(x))
    df[['prediction', 'probability']] = df['embedding'].apply(lambda x: pd.Series(predict_review_label(x)))
    print("Done prediction")
    return df

def visualize(df, filename):
    # Create a directory for storing plots specific to this file
    plot_dir = os.path.join("static/plots", filename)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Pie Chart
    pie_chart_path = percentage(df, plot_dir)

    # Word Clouds
    wordcloud_paths = get_wordcloud(df, plot_dir)

    # Trend Plot
    trend_plot_path = trend(df, plot_dir)

    # Get top positive and negative tweets
    positive_top2, negative_top2 = get_high_score(df)

    return pie_chart_path, wordcloud_paths, trend_plot_path, positive_top2, negative_top2



def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords_En])   
    return text

def predict_review_label(embedded_review):
    # embedded_review = np.array(embedded_review) #not sure??
    # prediction_prob = model.predict(embedded_review)
    prediction_prob = model.predict(embedded_review.reshape(1, -1))
    label = "Positive" if prediction_prob[0] > 0.5 else "Negative"
    return label, prediction_prob[0][0]

def get_wordcloud(df, plot_dir):
    # Word cloud for all tweets
    all_text = " ".join(df['cleaned_Text'])
    wordcloud_all = WordCloud(stopwords=stopwords_En, background_color="white").generate(all_text)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud_all, interpolation='bilinear')
    plt.axis('off')
    wordcloud_all_path = os.path.join(plot_dir, 'wordcloud_all.png')
    plt.savefig(wordcloud_all_path)
    plt.close()

    # Word cloud for positive tweets
    positive_text = " ".join(df[df['prediction'] == 'Positive']['cleaned_Text'])
    wordcloud_positive = WordCloud(stopwords=stopwords_En, background_color="white", colormap='Blues').generate(positive_text)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis('off')
    wordcloud_positive_path = os.path.join(plot_dir, 'wordcloud_positive.png')
    plt.savefig(wordcloud_positive_path)
    plt.close()

    # Word cloud for negative tweets
    negative_text = " ".join(df[df['prediction'] == 'Negative']['cleaned_Text'])
    wordcloud_negative = WordCloud(stopwords=stopwords_En, background_color="white", colormap='Reds').generate(negative_text)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis('off')
    wordcloud_negative_path = os.path.join(plot_dir, 'wordcloud_negative.png')
    plt.savefig(wordcloud_negative_path)
    plt.close()

    return wordcloud_all_path, wordcloud_positive_path, wordcloud_negative_path


def trend(df, plot_dir):
    # Resample data to every 12 hours and count positive and negative occurrences
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df = df.sort_values(by='time')
    df_resampled = df.resample('12H', on='time')['prediction'].value_counts().unstack().fillna(0)

    # Plotting the trend
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, df_resampled['Positive'], marker='o', color='b', label='Positive')
    plt.plot(df_resampled.index, df_resampled['Negative'], marker='o', color='r', label='Negative')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Positive and Negative Trends Over Time (12-hour intervals)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    trend_plot_path = os.path.join(plot_dir, 'trend_plot.png')
    plt.savefig(trend_plot_path)
    plt.close()

    return trend_plot_path


def get_high_score(df):
    # Sort by 'probability' in descending order
    sorted_data = df.sort_values(by='probability', ascending=False)

    # Get top 2 positive and top 2 negative tweets
    positive_top2 = sorted_data[sorted_data['prediction'] == 'Positive'].head(2)['text'].tolist()
    negative_top2 = sorted_data[sorted_data['prediction'] == 'Negative'].head(2)['text'].tolist()
    
    return positive_top2,negative_top2


def percentage(df, plot_dir):
    sentiment_counts = df['prediction'].value_counts()

    # Plot Pie chart
    plt.figure(figsize=(5, 5))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    pie_chart_path = os.path.join(plot_dir, 'pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()

    return pie_chart_path


