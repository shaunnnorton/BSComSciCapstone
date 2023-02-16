from flask import Blueprint, request, render_template, redirect,url_for, flash

from joblib import load
import pandas as pd
from math import floor


data = load("./Models/0004.joblib")
model = data["model"] 


main = Blueprint("main", __name__)

@main.route("/")
def mainRoute():
    return render_template("main.html")

@main.route("/predict")
def predictRoute():
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
    print(request.args.values())
    for i in request.args:
        if request.args.get(i) == "":
            return redirect(url_for("main.mainRoute"))
        
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
            
    startCodec = request.args.get("codec")
    transcodeCodec = request.args.get("output-codec")
    match startCodec:
        case "flv":
            data["onehotencoder__codec_flv"] = 1
        case "h264":
            data["onehotencoder__codec_h264"] = 1
        case "mpeg4":
            data["onehotencoder__codec_mpeg4"] = 1
        case "vp8":
            data["onehotencoder__codec_vp8"] = 1

    match transcodeCodec:
        case "flv":
            data["onehotencoder__o_codec_flv"] = 1
        case "h264":
            data["onehotencoder__o_codec_h264"] = 1
        case "mpeg4":
            data["onehotencoder__o_codec_mpeg4"] = 1
        case "vp8":
            data["onehotencoder__o_codec_vp8"] = 1

    data["remainder__frames"] =  floor(data['remainder__duration'] * data['remainder__framerate'])
    



    
    df = pd.DataFrame(data,index=[0]) 
    model.predict(df)   
    flash(f"It should take about {model.predict(df)[0]}seconds to transcode this video!!")
    return redirect(url_for("main.mainRoute"))

@main.route("/Model")
def modelRoute():
    return render_template("model.html")
