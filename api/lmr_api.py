# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import traceback

from flask import Flask, jsonify, request, send_file, send_from_directory, after_this_request, render_template, request
from flask_restx import Api, Resource
from flask_restx import fields, marshal_with

# from .twitter import TwitterSearch
from .analysis import MicroBlogAnalysis
# from .map import LocationMapping
from .data_processor import *
from .train import *
from .api_utils import write_api_logs

app = Flask(__name__)
api = Api(app)

#
# auth_schema_mandatory = api.model("AuthModel", {
#     "username": fields.String(required=True, description='UserName for the authentication.'),
#     "password": fields.String(required=True, descrition='Password for the authentication')
# })
#
#
# class FileMigration(Resource):
#     """
#     This provide file object after successful authentication.
#     """
#     auth_username = "ElliotVistaCedarWebCitrix"
#     auth_password = "aws"
#     dir_path = "/home/innefu/Documents/PROJECT/INNEFU/FlaskProjects/scrapy_cluster"
#
#     def get(self):
#         return {"status": "success",
#                 "message": "This api serves the purpose for migrating "
#                            "file only if Authentication successful & any file persist"}, 200
#
#     @api.expect(auth_schema_mandatory)
#     def post(self, **kwargs):
#         try:
#             response = api.payload
#             username = response['username']
#             password = response['password']
#             if username == FileMigration.auth_username and password == FileMigration.auth_password:
#                 return {
#                     "status": "success",
#                     "message": "login success!!!"
#                 }
#             else:
#                 if username != FileMigration.auth_username:
#                     error_username_msg = 'Incorrect Username'
#                 else:
#                     error_username_msg = ''
#                 if password != FileMigration.auth_password:
#                     error_password_msg = 'Incorrect Password'
#                 else:
#                     error_password_msg = ''
#                 return {
#                     "status": "error",
#                     "message": f"Please check your follow credentials: {error_username_msg} {error_password_msg}"}, 202
#         except Exception as e:
#             return {"status": "error",
#                     "message": f"Exited with error: {e}"}, 204
#
#
# keywords_search_schema_mandatory = api.model("AuthModel", {
#     "keywords": fields.String(required=True, description='Keywords to search on twitter for LMR.'),
# })
#
#
# class TwitterSearchLMR(Resource):
#
#     def get(self):
#         return {"status": "success",
#                 "message": "This api serves the purpose for searching & analysing tweets of a particular keyword/s for LMR"}, 200
#
#     @api.expect(keywords_search_schema_mandatory)
#     def post(self, **kwargs):
#         try:
#             response = api.payload
#             keywords = response["keywords"]
#             if len(keywords) == 0:
#                 return {"status": "error",
#                         "message": "Invalid input. Either blank or invalid words are sent as input."}, 202
#             twitter_search_result = TwitterSearch().process_tweets(keywords)
#             return {
#                 "status": "success",
#                 "message": f"{len(twitter_search_result)} tweets in result.",
#                 "tweets_data": twitter_search_result
#             }, 200
#         except Exception as e:
#             return {"status": "error",
#                     "message": f"Exited with error: {e}"}, 204
#
#
# text_schema_mandatory = api.model("AuthModel", {
#     "id": fields.Integer(required=True, description='Unique id for the mapping.'),
#     "text": fields.String(required=True, description='Keywords to search on twitter for LMR.'),
# })
#
#
# class TextLMR(Resource):
#
#     def get(self):
#         return {"status": "success",
#                 "message": "This api serves the purpose for analysing single text for LMR"}, 200
#
#     @api.expect(text_schema_mandatory)
#     def post(self, **kwargs):
#         try:
#             response = api.payload
#             ids = response["id"]
#             text = response["text"]
#             return {
#                     "tweet_id": ids,
#                     "text": text,
#                     "location_mentions": MicroBlogAnalysis().locations(text, ids)
#                 }, 200
#         except Exception as e:
#             print(traceback.print_exc())
#             return {"status": "error",
#                     "message": f"Exited with error: {e}"}, 204
#
#
# @app.route("/home", methods=['GET', 'POST'])
# def home():
#
#     def get_location_mentions(tweet):
#         return tweet['location_mentions']['location_mentions']
#
#     if request.method == 'POST':
#         text = request.form.get("search")
#         is_twitter = request.form.get("twitter")
#         print(text, is_twitter)
#
#         if is_twitter:
#             tweets = TwitterSearch().process_tweets(text)
#             print("<<<<<<<<<<", tweets)
#             location_mentions = list(map(get_location_mentions, tweets))
#             data = {
#                 "is_twitter": "yes",
#                 "keyword": text,
#                 "tweets_data": tweets,
#                 "map": LocationMapping().plot(prediction_data=location_mentions)
#             }
#             return render_template("pages/home.html", data=data)
#         else:
#             print(is_twitter, '0000000000')
#             tweets = MicroBlogAnalysis().locations(text, 1)
#             print(">>>>>", tweets)
#             data = {
#                 "is_twitter": "no",
#                 "tweet_id": tweets["tweet_id"],
#                 "location_mentions": tweets["location_mentions"],
#                 "map": LocationMapping().plot(prediction_data=tweets["location_mentions"])
#             }
#             return render_template("pages/home.html", data=data)
#     data = {
#         "is_twitter": "hmm",
#     }
#     return render_template("pages/home.html", data=data)
#
#
# @app.route("/chemical", methods=['GET', 'POST'])
# def chemical():
#
#     def get_location_mentions(tweet):
#         return tweet['location_mentions']['location_mentions']
#
#     if request.method == 'POST':
#         text = request.form.get("search")
#         is_twitter = request.form.get("twitter")
#         print(text, is_twitter)
#
#         if is_twitter:
#             tweets = TwitterSearch().process_tweets(text, processor='chemical')
#             print("<<<<<<<<<<", tweets)
#             location_mentions = list(map(get_location_mentions, tweets))
#             data = {
#                 "is_twitter": "yes",
#                 "keyword": text,
#                 "tweets_data": tweets,
#                 "map": None
#             }
#             return render_template("pages/chemical.html", data=data)
#         else:
#             print(is_twitter, '0000000000')
#             tweets = MicroBlogAnalysis().chemical(text, 1)
#             print(">>>>>", tweets)
#             data = {
#                 "is_twitter": "no",
#                 "tweet_id": tweets["tweet_id"],
#                 "location_mentions": tweets["location_mentions"],
#                 "map": None
#             }
#             return render_template("pages/chemical.html", data=data)
#     data = {
#         "is_twitter": "hmm",
#     }
#     return render_template("pages/chemical.html", data=data)
#
#
# @app.route("/fashion", methods=['GET', 'POST'])
# def fashion():
#
#     def get_location_mentions(tweet):
#         return tweet['location_mentions']['location_mentions']
#
#     if request.method == 'POST':
#         text = request.form.get("search")
#         is_twitter = request.form.get("twitter")
#         print(text, is_twitter)
#
#         if is_twitter:
#             tweets = TwitterSearch().process_tweets(text, processor='fashion')
#             print("<<<<<<<<<<", tweets)
#             location_mentions = list(map(get_location_mentions, tweets))
#             data = {
#                 "is_twitter": "yes",
#                 "keyword": text,
#                 "tweets_data": tweets,
#                 "map": None
#             }
#             return render_template("pages/fashion.html", data=data)
#         else:
#             print(is_twitter, '0000000000')
#             tweets = MicroBlogAnalysis().fashion(text, 1)
#             print(">>>>>", tweets)
#             data = {
#                 "is_twitter": "no",
#                 "tweet_id": tweets["tweet_id"],
#                 "location_mentions": tweets["location_mentions"],
#                 "map": None
#             }
#             return render_template("pages/fashion.html", data=data)
#     data = {
#         "is_twitter": "hmm",
#     }
#     return render_template("pages/fashion.html", data=data)
#
#
# @app.route("/organisation", methods=['GET', 'POST'])
# def organisation():
#
#     def get_location_mentions(tweet):
#         return tweet['location_mentions']['location_mentions']
#
#     if request.method == 'POST':
#         text = request.form.get("search")
#         is_twitter = request.form.get("twitter")
#         print(text, is_twitter)
#
#         if is_twitter:
#             tweets = TwitterSearch().process_tweets(text, processor='organization')
#             print("<<<<<<<<<<", tweets)
#             location_mentions = list(map(get_location_mentions, tweets))
#             data = {
#                 "is_twitter": "yes",
#                 "keyword": text,
#                 "tweets_data": tweets,
#                 "map": None
#             }
#             return render_template("pages/organization.html", data=data)
#         else:
#             print(is_twitter, '0000000000')
#             tweets = MicroBlogAnalysis().organization(text, 1)
#             print(">>>>>", tweets)
#             data = {
#                 "is_twitter": "no",
#                 "tweet_id": tweets["tweet_id"],
#                 "location_mentions": tweets["location_mentions"],
#                 "map": None
#             }
#             return render_template("pages/organization.html", data=data)
#     data = {
#         "is_twitter": "hmm",
#     }
#     return render_template("pages/organization.html", data=data)
#
#
# @app.route("/person", methods=['GET', 'POST'])
# def person():
#
#     def get_location_mentions(tweet):
#         return tweet['location_mentions']['location_mentions']
#
#     if request.method == 'POST':
#         text = request.form.get("search")
#         is_twitter = request.form.get("twitter")
#         print(text, is_twitter)
#
#         if is_twitter:
#             tweets = TwitterSearch().process_tweets(text, processor='person')
#             print("<<<<<<<<<<", tweets)
#             location_mentions = list(map(get_location_mentions, tweets))
#             data = {
#                 "is_twitter": "yes",
#                 "keyword": text,
#                 "tweets_data": tweets,
#                 "map": None
#             }
#             return render_template("pages/person.html", data=data)
#         else:
#             print(is_twitter, '0000000000')
#             tweets = MicroBlogAnalysis().persons(text, 1)
#             print(">>>>>", tweets)
#             data = {
#                 "is_twitter": "no",
#                 "tweet_id": tweets["tweet_id"],
#                 "location_mentions": tweets["location_mentions"],
#                 "map": None
#             }
#             return render_template("pages/person.html", data=data)
#     data = {
#         "is_twitter": "hmm",
#     }
#     return render_template("pages/person.html", data=data)
#
#
# @app.route("/disease", methods=['GET', 'POST'])
# def disease():
#
#     def get_location_mentions(tweet):
#         return tweet['location_mentions']['location_mentions']
#
#     if request.method == 'POST':
#         text = request.form.get("search")
#         is_twitter = request.form.get("twitter")
#         print(text, is_twitter)
#
#         if is_twitter:
#             tweets = TwitterSearch().process_tweets(text, processor='disease')
#             print("<<<<<<<<<<", tweets)
#             location_mentions = list(map(get_location_mentions, tweets))
#             data = {
#                 "is_twitter": "yes",
#                 "keyword": text,
#                 "tweets_data": tweets,
#                 "map": None,
#             }
#             return render_template("pages/disease.html", data=data)
#         else:
#             print(is_twitter, '0000000000')
#             tweets = MicroBlogAnalysis().disease(text, 1)
#             print(">>>>>", tweets)
#             data = {
#                 "is_twitter": "no",
#                 "tweet_id": tweets["tweet_id"],
#                 "location_mentions": tweets["location_mentions"],
#                 "map": None,
#             }
#             return render_template("pages/disease.html", data=data)
#     data = {
#         "is_twitter": "hmm",
#     }
#     return render_template("pages/disease.html", data=data)


# ################################ DOCKER USER BUILT ###########################################

@app.route("/main", methods=['GET', 'POST'])
def main():

    if request.method == 'POST':
        try:
            text = request.form.get("text")
            model = request.form.get("model")
            # is_posam = request.form.get("is_posam")
            print(text, model)
            # tweets = MicroBlogAnalysis().persons(text, 1)
            if 'mdm' in model or 'vsm' in model:
                multi_embed=True
            else:
                multi_embed = False
            tweets = MicroBlogAnalysis().locations(text, 1, model, multi_embed)

            data = {
                "is_twitter": "no",
                "tweet_id": tweets["tweet_id"],
                "location_mentions": tweets["location_mentions"],
                "map": None,
            }
            print('>>>>>', data)
            return render_template("pages/results.html", data=data)
        except Exception as e:
            data = {
                "message": str(e)
            }
            return render_template("pages/error.html", data=data)
    data = {
        "is_twitter": "hmm",
    }
    return render_template("pages/main.html", data=data)


@app.route("/training", methods=['GET', 'POST'])
def training():

    if request.method == 'POST':
        try:
            model = request.form.get("model")
            # lr = request.form.get("learning_rate")
            target_class = request.form.get("target")
            # training_type = request.form.get("training_type")
            print(model, target_class)
            write_api_logs(f"Received Input-\nModel:{model}\nTarget Class:{target_class}")
            da = DatasetAnalysis()
            report = da.report(target_class)
            report['target_class'] = target_class

            data_processor = SaveDataset().save(target_class=target_class, experiment_name=model)
            learning = ExperimentalLearning()
            learning.reduce_learning()

            data = {'logs': []}
            fls = open(api_log_file)

            for log in fls:
                data['logs'].append(log.strip())
            return render_template("pages/logs.html", data=data)
        except Exception as e:
            data = {
                "message": str(e)
            }
            return render_template("pages/error.html", data=data)
    data = {
        "is_twitter": "hmm",
    }
    return render_template("pages/main.html", data=data)


@app.route("/report", methods=['GET', 'POST'])
def report():

    if request.method == 'POST':
        pass
    try:
        data = {'logs':[]}
        fls = open(api_log_file)

        for log in fls:
            data['logs'].append(log.strip())
    except Exception as e:
        data = {
            "message": str(e)
        }
        return render_template("pages/error.html", data=data)

    return render_template("pages/logs.html", data=data)


@app.route("/readme", methods=['GET', 'POST'])
def readme():

    if request.method == 'POST':
        pass

    return render_template("pages/readme.html", data={})


# api.add_resource(FileMigration, '/file_migrations')
# api.add_resource(TwitterSearchLMR, '/twitter_search')
# api.add_resource(TextLMR, '/simple_search')

if __name__ == "__main__":
    os.system("set FLASK_APP=LMR_api")
    app.run(host="0.0.0.0", port="5000", debug=True)