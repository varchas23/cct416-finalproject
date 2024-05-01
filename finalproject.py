"""Reddit Communities’ Sentiments On the Apple Vision Pro
Varchas Sharma: 1007112731
Ismail Mostafa: 1006321437
Yuwen Lin: 1007258628
CCT416H5
"""

# Importing libraries
import praw, pandas as pd, matplotlib.pyplot as plt, nltk, pyLDAvis.gensim
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models import LdaModel
from matplotlib.backends.backend_pdf import PdfPages
from nltk.sentiment import SentimentIntensityAnalyzer

"""Class that will scrape data from Reddit and create visualizations based on LDA Topic Modelling
LDA Topic Modelling and Sentiment Analysis"""
class ScrapeReddit:
    # Initializer method to take in Reddit API, the desired subreddit and set up dataframe
    def __init__(self, subreddit):
        self.reddit = praw.Reddit(client_id='T-BzUYy74qBZAwx-L6thlg',
                     client_secret='e6JB8aTEqu4NL9M4cUJqUx_9n7tlHQ',
                     user_agent='MyRedditScraper by /u/varchas23')
        self.subreddit = subreddit
        self.df = pd.DataFrame()
    
    # Scrapes top 100 posts from subreddit and gets added to dataframe
    def scrape_subreddits(self):
        sub = self.reddit.subreddit(self.subreddit)
        top = sub.top(limit=100)
        data = []
        for posts in top:
            post_data = {
                "Title": posts.title,
                "Score": posts.score,
                "Content": posts.selftext if posts.selftext else None,
                "Number of Comments": posts.num_comments,
                #"Author": post.author.name if post.author else None,
                "Post URL": posts.url,
                "Full URL": f"https://www.reddit.com{posts.permalink}"
            }
            data.append(post_data)
        self.df = pd.DataFrame(data)
        print(self.df)
        print(f'The minimum score is: {min(self.df["Score"])}\nThe maximum score is: {max(self.df["Score"])} \
              \nThe minimum number of comments is: {min(self.df["Number of Comments"])}\nThe maximum number of comments\
              is: {max(self.df["Number of Comments"])}\nThe average score is: {sum(self.df["Score"]) / len(self.df["Score"])} \
                \nThe average number of comments is: {sum(self.df["Number of Comments"]) / len(self.df["Number of Comments"])}')
        return self.df
    
    # Counts number of words per post and for all posts, then finds the average number of words
    def count_words(self):
        count, count_lst = 0, []
        for post in self.df['Content']:
            words = word_tokenize(str(post))
            count_lst.append(len(words))
            count += len(words)
        average = count / len(count_lst)
        print(f'There are {count} words in all the posts. \
The average number of words in each post is {average}')
    
    # Method that cleans the data using tokenization and stemming
    def preprocess_data(self, content):
        if isinstance(content, str):
            tokens = word_tokenize(content)
            tokens = [token.lower() for token in tokens if token.isalnum()]
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            porter = PorterStemmer()
            stemmed_tokens = [porter.stem(word) for word in tokens]
            return stemmed_tokens
        else:
            return []

    # Topic Modelling using LDA to collect and categorize words in specific topics that are related
    # Visualization with bar graphs and Intertopic Distance Map
    def topic_modelling(self):
        self.df['preprocessed_posts'] = self.df['Content'].apply(self.preprocess_data)
        dictionary = corpora.Dictionary(self.df['preprocessed_posts'])
        corpus = [dictionary.doc2bow(tokens) for tokens in self.df['preprocessed_posts']]
        num_topics = 10
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=100)
        topics = lda.show_topics(num_topics=num_topics, num_words=10)
        for topic in topics:
            print(topic)

        n_top_words = 5
        with PdfPages('lda_model_topics.pdf') as pdf:
            plt.figure(figsize=(10, 10))
            for topic_idx, topic in lda.show_topics(num_topics=num_topics, num_words=n_top_words, formatted=False):
                top_words = [word for word, _ in topic]
                top_word_probs = [prob for _, prob in topic]

                plt.subplot(5, 2, topic_idx + 1)
                plt.barh(top_words, top_word_probs, color='blue', edgecolor='black')
                plt.gca().invert_yaxis()
                plt.title(f'Topic {topic_idx + 1}', fontsize=12)

            plt.tight_layout()
            pdf.savefig()
            plt.show()
        vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(vis_data, f'{self.subreddit}lda_visualization.html')

    # Helper function to determine sentiment of data
    def categorize_sentiment(self, sentiment):
        compound = sentiment['compound']
        if compound < 0:
            return 'Negative'
        elif compound == 0:
            return 'Neutral'
        else:
            return 'Positive'

    # Applies sentiment analysis to title of each post with positive, negative or neutral sentiment
    # Visualization using scatter graph displaying number of comments and score
    def sentiment_analysis(self):
        sid = SentimentIntensityAnalyzer()
        self.df['sentiment_score'] = self.df['Title'].apply(lambda x: sid.polarity_scores(x))
        self.df['sentiment_category'] = self.df['sentiment_score'].apply(self.categorize_sentiment)

        colors = {'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}

        plt.figure(figsize=(10, 6))

        for sentiment_category, color in colors.items():
            subset_df = self.df[self.df['sentiment_category'] == sentiment_category]
            plt.scatter(subset_df['Number of Comments'], subset_df['Score'], color=color, label=sentiment_category, alpha=0.5)

        plt.title('Relationship Between Number of Comments and Score by Sentiment')
        plt.xlabel('Number of Comments')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()

        plt.show()

# Runs the code
if __name__ == "__main__":
    # Stores each subreddit into list
    subreddit = ['AppleVisionPro', 'VisionPro', 'AppleVision', 'apple']
    # For loop to iterate through each subreddit
    for i in subreddit:
        sub = ScrapeReddit(i)
        sub.scrape_subreddits()
        sub.count_words()
        sub.topic_modelling()
        sub.sentiment_analysis()