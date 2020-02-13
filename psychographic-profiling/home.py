from flask import Flask, render_template, request
import tweepy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

## it's referencing this file
app = Flask(__name__,template_folder='template')

#define route for that route
@app.route('/')
def run_app():
    return render_template('index.html')

def get_credentials():
    #Add your credentials here
    twitter_keys = {
            'consumer_key':        '3HcSvDCAMt8IJc4PG3nqrLA3d',
            'consumer_secret':     '97GGmrvQKsgA9xFCppFAk9x4dJycRENlZSTeujvkCxBztofhyT',
            'access_token_key':    '1059862055564533760-c0CLEY9gpXJ74y4y7gvmnBLfknTOiG',
            'access_token_secret': 'Q9R0KSOjwXUBkdu1QTJdxYTeH3rR5QNPQIJZrytt3IQkH'
        }

    #Setup access to API
    auth = tweepy.OAuthHandler(twitter_keys['consumer_key'], twitter_keys['consumer_secret'])
    auth.set_access_token(twitter_keys['access_token_key'], twitter_keys['access_token_secret'])
    api = tweepy.API(auth)
    return api

api = get_credentials()

@app.route('/', methods = ['POST'])

def get_user_description():
    try:
        request_input = request.form['twitterUsername']
        user = api.get_user(request_input)
        name = user.name
        description = user.description
        string = ""
        if description == string:
            user = 'User does not have a description.'
        else:
            user = [description]
            filename = 'finalized_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            loaded_vec = TfidfVectorizer(vocabulary=pickle.load(open("feature.pkl", "rb")))
            unseen_tfidf = loaded_vec.fit_transform(user)
            results = loaded_model.predict(unseen_tfidf)[0]
            if results == 0:
                user = 'Hi {}!! You are likely to donate to charities!!'.format(name)
            else:
                user = 'Hi {} !! You are not likely to donate to charities :('.format(name)
    except:
        user = 'This user does not exist. Try again!'
    return render_template('output.html', a_user=user, name = name)

if __name__ == '__main__':
    app.debug = True
    app.run()

