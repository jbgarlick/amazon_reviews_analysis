import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Title, header, and contact information
st.title('Sentiment Analysis of Amazon Reviews')
st.subheader('By: Jared Garlick')
st.write('jared.b.garlick@gmail.com')
st.write('[jaredgarlick.com](https://jaredgarlick.com)')
st.write('[LinkedIn](https://www.linkedin.com/in/jaredgarlick/)')
st.write('[Github](https://github.com/jbgarlick)')

# Introduction
st.subheader('Introduction')
st.write('Have you ever looked at something to buy on Amazon, or some other e-commerce website, and immediately check the reviews? You\'re not alone. Most people trust the \
        reviews more than anything when they are considering an online purchase. According to a recent study, 9 out of 10 people read reviews before purchasing something, and \
        54.7% of online consumers read at least four reviews before purchasing. Personally, unless it\'s something that I already know about and want, I would never buy something \
        online without first checking out the reviews. But I don\'t want to have to read every review before I can decide whether I want something or not. \
        Sure, the star ratings can be helpful, but it\'s often the case that a scale from 1 to 5 can\'t quite describe exactly how someone feels.')

st.write('Consider an example of this with a review for a Cheese Cloth on Amazon. The top rated review is titled **\'Quality must have for the kitchen!\'**, with pictures \
        and a paragraph of praise, ultimately given a rating of _5 stars_. But upon reading the reivew, she mentions:')
st.text('These are susceptible to staining though and especially if you\'re using any plant-based material it\'s going to change the color of the product after you use it even for the first time.')

st.write('To me, this doesn\'t feel like a 5 star review, but it also doesn\'t feel like a 4 star review either. Maybe 4.8, or somewhere in that range. The trouble with having a discrete \
        scale for ratings is that people have to round up or round down based on how they\'re feeling, which usually involves some positive and some negative emotion.')

st.write('The goal for this project is to utilize Sentiment Analysis and other Natural Language Proccessing tools and techniques to compare the star rating of a review, and compare \
        that to how positive or negative the sentiment of the review was, giving it a score that can be used as a different measure for rating.')

# Summary of previous or related approaches
st.subheader('Related work and research')
st.write('[Ranking Aspect-Based Features in Restaurant Reviews](https://scholarsarchive.byu.edu/etd/8733/) by Jacob Ling Hang Chan, Brigham Young University')
st.write('[The emergence of social media data and sentiment analysis in election prediction](https://link.springer.com/article/10.1007/s12652-020-02423-y) by Priyavrat Chauhan, Nonita Sharma & Geeta Sikka ')
st.write('[Deep Learning for Aspect-Based Sentiment Analysis: A Comparative Review](https://www.sciencedirect.com/science/article/abs/pii/S0957417418306456) by Hai HaDoP, WCPrasad, AngelikaMaag, AbeerAlsadoon')

# Summary of corpora, langauges used, and/or toolkits used
st.subheader('Data Sources and Tools Used')
st.write('For this project, I used a corpus of 200,000 amazon reviews from various shopping departments, and evenly distributed star ratings, 40,000 each. The data is available to download here as well.')
st.write('I used Python to code this project, and the NLTK toolkit.')
st.download_button('Download Amazon Reviews Corpus', 'data_train.json', 'all_amazon_reviews.json')

# import the data
df = pd.read_csv('all_data.csv')

# Analyze each review, show that it's inconclusive/gives evenly distributed result, select variables to plot
st.subheader('First Iteration')
st.write('To begin, I ran the NLTK sentiment analysis algorithm on each review text and got back the polarity scores for each. I will focus on the compound score, which is scored \
        on a scale from -1 to 1, with scores closer to 1 having more positive sentiment, and vice versa.')
st.write('The results from this are somewhat inconclusive, given that each star rating has polarity scores that range from -1 to 1, with no real pattern to suggest correlation between the two, \
        as you can see in the scatter plot below:')
fig = px.scatter(df, x='stars', y='compound')
st.plotly_chart(fig)

st.write('But, upon futher investigation, the average of these scores actually beings to paint the picture we were hoping for. Notice the average compound score for each: ')
st.text('5 star reviews: 0.65')
st.text('4 star reviews: 0.52')
st.text('3 star reviews: 0.22')
st.text('2 star reviews: 0.03')
st.text('1 star reviews: -0.15')
st.write('So, on average the stars given in a review correlate pretty well with the positive or negative sentiment in the review body.')


# Then break down each review into each sentence
# Run sentiment analysis on each of those sentences, get the average score or if it's mostly positive or negative, compare these results, select variables to plot
st.subheader('Second Iteration')
st.write('I wanted to take it a step further and rather than looking at the review as a whole, I wanted to look at each sentence individually, and then come up with a combinatory score based on \
        the sentiment of each sentence. Based on the example I shared above, this should help us take into consideration parts of those 5 star reviews that should bring it down closer to a 4 star review.')
st.write('My algorithm used this code to break up the paragraph into a list of each sentence:')
st.code('sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", review)')
st.write('Then, for each of those sentences, I ran the NLTK sentiment analysis algorithm and got its score. I took the average of each of the sentences compound score, and stored that as \
        the review\'s new score.')
st.write('The results for this method were interesting. They followed a similar pattern as before, but were each on average less positive than the respective star rating when looking at the whole review.')
st.text('5 star reviews: 0.44')
st.text('4 star reviews: 0.31')
st.text('3 star reviews: 0.11')
st.text('2 star reviews: 0.0003')
st.text('1 star reviews: -0.09')
st.write('This is evidence that most positive reviews contain some negative parts to them, and not all negative reviews only contain negative feedback. \
        When looking at the entire review, 4 star reviews had an average score of 0.52, but when looking at each sentence and getting the average score, most 5 star reviews had 0.44, \
        _less than the amount of 4 stars previously_.')

st.subheader('Third Iteration')
st.write('Lastly, I thought the average could have some issues given that most sentences are neutral when scored, as well as not wanting to bring the average down too much for a solid 5 star review that might have one little complaint in it.')
st.write('To accomplish this, I had a similar method as the Second Iteration, breaking up the review into each sentence, but instead of taking the average, I took the sentiment that \
        had the majority. For example, let\'s say a review had 5 sentences, 4 of them being positive and 1 of them being negative. I considered the overall tone of this review to be positive. \
        Likewise, if a review had 7 sentences, 3 negative, 2 positive, and 2 neutral, I considered the overall tone of this review to be negative. If there was a tie for the majority, I considered it \
        positive\negative or negative\neutral, a combination of two, and so on. If they all tied, that would be considered completely neutral.')
st.write('The results of this method were pretty fascinating, as seen in the table below:')
st.image('final_table_pic.png')
st.write('By this method, a large percentage of the reviews are considered positive, but those still average a little under 4 stars.')
st.write('Most intersting to me was that the overall tones followed the average stars almost exactly. This means that the method did a good job of predicting how many stars a review might have. \
        The most positive reviews will typically have the most stars, the most negative reviews will have the least amount of stars, and the most neutral will stay right in the middle. \
        This result leads me to believe that if you see an item you are wanting to purchase, and it has a lot of 3, 4, or 5 star reviews, you can trust that the response is pretty positive. \
        You don\'t need to only look for items with 5,000 5 star reviews, and nothing else. Also don\'t be immediately turned off by the 3 or 4 star reviews. They can often be pretty positive, with maybe \
        one or two things they didn\'t like about the product, which would be good to know because that might not be an issue for you.')

st.subheader('Fourth Iteration')
st.write('Finally I want the user to get a hands-on experience with how these results might change based on the category an item is in.')
st.write('Select a category from the drop down box below, and you will see the average compound scores for each star rating. What jumps out at you? Notice anything interesting? \
        Which categories would you expect to have the most positive reviews? How about most negative?')
categories = df['category'].to_list()
categories = set(categories)
categories = sorted(categories)
option = st.selectbox('Select a Product Category: ', ['Select a Category'] + list(categories))

if option != 'Select a Category':
    df = df[df['category'] == option]
    df = df[['stars', 'compound', 'avg compound', 'general tone']]
    st.table(df.groupby('stars')['compound'].mean())

st.subheader('Conclusion')
st.write('Sentiment Analysis is a fascinating way to apply Natural Language Processing Techniques to the real world. Emotion is something that we all feel and understand, but \
        it is difficult for a machine to be able to understand that emotion. In this project, I focused on positivity, negativity, and neutrality. There are so many \
        more emotions that we feel and write about and talk about, and the research being done to have a computer process and understand all of that emotion is pretty awesome! \
        I hope you learned something beacuse of my project, or that you try to learn more after this. If anything I hope you\'ll read those Amazon reviews a little closer next \
        time you\'re shopping and think about how the computer might interpret those reviews.')




# Sources 
st.subheader('Sources')
st.text("https://www.oberlo.com/blog/online-review-statistics#:~:text=Most%20consumers%20who%20read%20online,(Bizrate%20Insights%2C%202021)")
st.write('')
st.write('')
st.write('')
st.write('')
st.text('Created by Jared Garlick')
st.write('[jaredgarlick.com](https://jaredgarlick.com)')
st.write('[LinkedIn](https://www.linkedin.com/in/jaredgarlick/)')
st.markdown('[Github](https://github.com/jbgarlick)')
st.subheader('Sources')
st.text("https://www.oberlo.com/blog/online-review-statistics#:~:text=Most%20consumers%20who%20read%20online,(Bizrate%20Insights%2C%202021)")