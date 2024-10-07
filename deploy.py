from flask import Flask, request, render_template
from pipelinefunctions import preprocess, prediction, visualize
from web_scraping import scrape_tweets

max_tweets=1
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        username = request.form['username']
        password = request.form['password']
        print('username:',username)
        print('password:',password)
        print('topic:',topic)
        
        tweets,filename = scrape_tweets(username, password, topic, max_tweets)

        processed_tweets = preprocess(tweets)

        sentiment_results = prediction(processed_tweets)

        pie_chart, wordcloud_paths, trend_plot, positive_top2, negative_top2 = visualize(sentiment_results, filename)
        return render_template('results.html', 
                               pie_chart=pie_chart, 
                               wordcloud_paths=wordcloud_paths, 
                               trend_plot=trend_plot, 
                               tweet_samples=(positive_top2, negative_top2))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)