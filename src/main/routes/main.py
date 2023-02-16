from flask import Blueprint, request, render_template, redirect,url_for, flash

from joblib import load
import pandas as pd
from math import floor

# Load the machine learning model to memory
data = load("./Models/0004.joblib")
model = data["model"] 


main = Blueprint("main", __name__)

@main.route("/")
def mainRoute():
    '''The main page of the webiste. Displays a form for calculating a transcoding time prediciton.'''
    return render_template("main.html")

@main.route("/predict")
def predictRoute():
    """Route to predict the transcoding time using the machine learning model. 
    Flashes a message with the time prediction and redirects back to the main route"""
    # #######################################################################
    # Sample data in the correct order to pass to the machine learning model
    # #######################################################################
    # data = {'onehotencoder__codec_flv':0,
    #         'onehotencoder__codec_h264':0,
    #         'onehotencoder__codec_mpeg4' :1,
    #         'onehotencoder__codec_vp8':0,
    #         'onehotencoder__o_codec_flv' :0,
    #         'onehotencoder__o_codec_h264':0,
    #         'onehotencoder__o_codec_mpeg4' :1,
    #         'onehotencoder__o_codec_vp8':0,
    #         'remainder__duration' :130.35667,
    #         'remainder__width' :176,
    #         'remainder__height':144,
    #         'remainder__bitrate' :54590,
    #         'remainder__framerate' :12,
    #         'remainder__frames':1564,
    #         'remainder__size' :889537,
    #         'remainder__o_bitrate' :56000,
    #         'remainder__o_framerate':12,
    #         'remainder__o_width':176,
    #         'remainder__o_height':144}
    
    # Check all arguements were provided in the route. 
    for i in request.args:
        if request.args.get(i) == "":
            return redirect(url_for("main.mainRoute"))

    # Init the data object    
    data = {
            'onehotencoder__codec_flv':0,
            'onehotencoder__codec_h264':0,
            'onehotencoder__codec_mpeg4' :0,
            'onehotencoder__codec_vp8':0,
            'onehotencoder__o_codec_flv' :0,
            'onehotencoder__o_codec_h264':0,
            'onehotencoder__o_codec_mpeg4' :0,
            'onehotencoder__o_codec_vp8':0,

            'remainder__duration' :float(request.args.get("duration")),
            'remainder__width' :float(request.args.get("video-width")),
            'remainder__height':float(request.args.get("video-height")),
            'remainder__bitrate' :float(request.args.get("video-bitrate")),
            'remainder__framerate' :float(request.args.get("video-framerate")),
            'remainder__frames': 0,
            'remainder__size' :float(request.args.get("video-size")),
            'remainder__o_bitrate' :float(request.args.get("output-video-bitrate")),
            'remainder__o_framerate':float(request.args.get("output-video-framerate")),
            'remainder__o_width':float(request.args.get("output-video-width")),
            'remainder__o_height':float(request.args.get("output-video-height"))
            }

    # Get the codec values passed in the URL        
    startCodec = request.args.get("codec")
    transcodeCodec = request.args.get("output-codec")
    # Set the start codec value to 1 based on the provided value
    match startCodec:
        case "flv":
            data["onehotencoder__codec_flv"] = 1
        case "h264":
            data["onehotencoder__codec_h264"] = 1
        case "mpeg4":
            data["onehotencoder__codec_mpeg4"] = 1
        case "vp8":
            data["onehotencoder__codec_vp8"] = 1

    # Set the transcode codec value to 1 based on the provided value
    match transcodeCodec:
        case "flv":
            data["onehotencoder__o_codec_flv"] = 1
        case "h264":
            data["onehotencoder__o_codec_h264"] = 1
        case "mpeg4":
            data["onehotencoder__o_codec_mpeg4"] = 1
        case "vp8":
            data["onehotencoder__o_codec_vp8"] = 1

    # Calculate and set the number of frames using the frame rate and duration
    data["remainder__frames"] =  floor(data['remainder__duration'] * data['remainder__framerate'])
    



    # Create the dataframe using the data provided
    df = pd.DataFrame(data,index=[0]) 
    # Flash the prediciton to the user and redirect to main page.    
    flash(f"It should take about {model.predict(df)[0]}seconds to transcode this video!!")
    return redirect(url_for("main.mainRoute"))

@main.route("/Model")
def modelRoute():
    """Route for infomation on the model"""
    return render_template("model.html")
