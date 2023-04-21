from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras.applications.resnet import preprocess_input
from tensorflow import keras
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, Response, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# from flask_ngrok import run_with_ngrok

from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# from keras.preprocessing import image


MODEL_PATH = 'model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = 'Bacterial_spot \r\n Copper sprays can be used to control bacterial leaf spot, but they are not as effective when used alone on a continuous basis'
    elif preds == 1:
        preds = "Early_blight \n Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable Spray Oil."
    elif preds == 2:
        preds = "Late_blight \n Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used"
    elif preds == 3:
        preds = "Leaf_Mold \n Remove and destroy all affected plant parts. For plants growing under cover, increase ventilation and, if possible, the space between plants. Try to avoid wetting the leaves when watering plants, especially when watering in the evening, Copper-based fungicides can be used to control diseases on tomatoes"
    elif preds == 4:
        preds = "Septoria leaf spot \n Remove diseased leaves.Improve air circulation around the plants.Do not use overhead watering.Use fungicidal sprays."
    elif preds == 5:
        preds = "Spider mites Two spotted spider mite \n Do not over-fertilize. Outbreaks may be worsened by excess nitrogen fertilization.Overhead irrigation or prolonged periods of rain can help reduce populations."
    elif preds == 6:
        preds = "Target Spot \n Warm wet conditions favour the disease such that fungicides are needed to give adequate control. The products to use are chlorothalonil, copper oxychloride or mancozeb"
    elif preds == 7:
        preds = "Yellow Leaf Curl Virus \n Once infected with the virus, there are no treatments against the infection. Control the whitefly population to avoid the infection with the virus. Insecticides of the family of the pyrethroids used as soil drenches or spray during the seedling stage can reduce the population of whiteflies."
    elif preds == 8:
        preds = "mosaic virus \n Remove any infected plants, including the roots, Also discard any plants near those affected. Like all viruses, mosaics are incurable- although sometimes they simply create interestingly patterned leaves without significantly reducing a plant's vigor"
    else:
        preds = "Healthy"
    return preds


app = Flask(__name__,template_folder='template')
app.secret_key = 'hello'

# starts ngrok when the app is run

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
db.init_app(app)

login_manger = LoginManager()
login_manger.init_app(app)


class Users(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(80))


class Reports(UserMixin, db.Model):
    __tablename__ = 'reports'
    report_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # week = db.Column(db.String(120),unique=True)
    # file_path = db.Column(db.String(1000))
    current_date = db.Column(db.Date)
    image = db.Column(db.LargeBinary)
    pred_val = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))


@login_manger.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        u_email = request.form["u_email"]
        u_pwd = request.form["u_pwd"]
        login = Users.query.filter_by(email=u_email, password=u_pwd).first()
        if login is not None:
            login_user(login)
            return redirect('/home')
        else:
            flash('Username or password is wrong')
            return redirect('/login')
    return render_template("login.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        _uname = request.form["uname"]
        _email = request.form["email"]
        _passw = request.form["pwd"]
        _cpassw = request.form["cpwd"]
        if _passw == _cpassw:
            register = Users(name=_uname, email=_email, password=_passw)
            db.session.add(register)
            db.session.commit()
            return redirect('/login')
        else:
            flash('Password and Confirm Password does not match')
            return redirect('/register')
    return render_template("register.html")


@app.route('/home')
@login_required
def home():
    return render_template('home.html', users=current_user)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        _user_id = current_user.id
        today = datetime.today()
        to_date = today.strftime("%Y-%m-%d")
        t_date = datetime.strptime(to_date, '%Y-%m-%d').date()
        convert_to_png(file_path)
        with open(file_path, "rb") as fi:
            binary_data = fi.read()
            report = Reports(current_date=t_date, image=binary_data,
                             pred_val=result, user_id=_user_id)
            db.session.add(report)
            db.session.commit()

        return result
    return render_template('home.html', result=result)


def convert_to_png(image_path):
    # Open the image file
    img = cv2.imread(image_path)
    # Save the image as a PNG file
    cv2.imwrite(image_path.replace(".jpg", ".png"), img)


@app.route('/image/<int:image_id>')
def image(image_id):
    reports = Reports.query.get(image_id)
    return send_file(BytesIO(reports.image), mimetype='image/png')


@app.route('/report', methods=['GET', 'POST'])
@login_required
def report():
    if request.method == "GET":
        _user_id = current_user.id
        report = Reports.query.filter_by(user_id=_user_id).all()
    return render_template("report.html", report=report, users=current_user)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


if __name__ == '__main__':
    app.run(port=8000, debug=True)
