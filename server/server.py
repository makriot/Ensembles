from flask import Flask, render_template
from flask import request, url_for, redirect
from flask_bootstrap import Bootstrap
from flask import session, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired

from wtforms.validators import DataRequired, ValidationError
from wtforms.validators import DataRequired
from wtforms.validators import StopValidation

from wtforms import FileField, SubmitField, StringField
from wtforms import IntegerField, FloatField

from math import isnan
import ensembles_web as ens
import pandas as pd
import os

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'snowden'
Bootstrap(app)

data_path = "./data"

model = None
train_df = None
val_df = None
default_param = -1


## checkers
class Number_checker(object):

    def __init__(self, min=None, max=None, message=None, greater=False):
        self.min = min
        self.max = max
        self.message = message
        self.greater = greater

    def __call__(self, form, field):
        data = field.data
        if data is not None and data == default_param:
            return
        if data is None or isnan(data) or (self.min is not None and data < self.min) or \
                (self.max is not None and data > self.max):
            message = self.message
            if message is None:
                if self.max is None:
                    if self.greater:
                        message = field.gettext('Number must be greater than %(min)s.')
                    else:
                        message = field.gettext('Number must be at least %(min)s.')
                elif self.min is None:
                    message = field.gettext('Number must be at most %(max)s.')
                else:
                    message = field.gettext('Number must be between %(min)s and %(max)s.')

            raise ValidationError(message % dict(min=self.min, max=self.max))


class Train_csv_data_checker(object):

    def __call__(self, form, field):
        global train_df
        try:
            stream = field.data.stream
            df = pd.read_csv(stream)
        except Exception:
            train_df = None
            raise ValidationError(field.gettext("Can't interpret file"))
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                train_df = df
                return
        train_df = None
        raise ValidationError(field.gettext("No numeric data in csv file"))


class Test_csv_data_checker(object):

    def __call__(self, form, field):
        global val_df
        try:
            stream = field.data.stream
            df = pd.read_csv(stream)
        except Exception:
            val_df = None
            raise ValidationError(field.gettext("Can't interpret file"))
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                val_df = df
                return
        val_df = None
        raise ValidationError(field.gettext("No numeric data in csv file"))
        

class Target_checker(object):

    def __call__(self, form, field):
        if train_df is None:
            raise ValidationError("No valid dataset is provided")
        if field.data not in train_df.columns:
            raise ValidationError(field.gettext("No such column: '%s'.") % field.data)


class File_optional(object):

    def __call__(self, form, field):
        if not field.data:
            field.errors[:] = []
            raise StopValidation()


## fields
class ChooseButton(FlaskForm):
    rf_model = SubmitField("Random Forest")
    gb_model = SubmitField("Gradient Boosting")


class Params_RF(FlaskForm):
    n_estimators = IntegerField("Number of trees", 
                validators=[DataRequired(), Number_checker(min=1)])
    max_depth = IntegerField("Maximum depth",
                validators=[DataRequired(), Number_checker(min=1)])
    feature_subsample_size = IntegerField("Number of features",
                validators=[DataRequired(), Number_checker(min=1)])
    train_file = FileField('Train dataset', validators=[
        FileRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!'),
        Train_csv_data_checker()
    ])
    target = StringField('Target label', 
                validators=[DataRequired(), Target_checker()])
    test_file = FileField('Validation dataset (optional)', validators=[
        File_optional(),
        FileAllowed(['csv'], 'CSV only!'),
        Test_csv_data_checker()
    ])
    button = SubmitField("Train")


class Params_GB(FlaskForm):
    n_estimators = IntegerField("Number of trees", 
                validators=[DataRequired(), Number_checker(min=1)])
    learning_rate = FloatField("Learning rate",
                validators=[DataRequired(), Number_checker(min=0, greater=True)])
    max_depth = IntegerField("Maximum depth",
                validators=[DataRequired(), Number_checker(min=1)])
    feature_subsample_size = IntegerField("Number of features",
                validators=[DataRequired(), Number_checker(min=1)])
    train_file = FileField('Train dataset', validators=[
        FileRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!'),
        Train_csv_data_checker()
    ])
    target = StringField('Target label', 
                validators=[DataRequired(), Target_checker()])
    test_file = FileField('Validation dataset (optional)', validators=[
        File_optional(),
        FileAllowed(['csv'], 'CSV only!'),
        Test_csv_data_checker()
    ])
    button = SubmitField("Train")


class Default_params(FlaskForm):
    button = SubmitField("Fill default parameters")


class Again_button(FlaskForm):
    button = SubmitField("Try again")


class Submission_button(FlaskForm):
    button = SubmitField("Predict")


class CSV_upload(FlaskForm):
    file_obj = FileField('Dataset', validators=[
        FileRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!'),
        Test_csv_data_checker()
    ])
    submit = SubmitField("Submit")



def training_model(model_type, params, X_train, y_train, X_val=None, y_val=None):
    global model
    if model_type == "RandomForest":
        model = ens.RandomForestMSE(**params)
    elif model_type == "GradientBoosting":
        model = ens.GradientBoostingMSE(**params)
    else:
        return None
    yield from model.fit(X_train, y_train, X_val, y_val)

def preprocess_data(df, target=None):
    col_list = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_list.append(col)
    if target is None:
        return df[col_list].to_numpy()
    else:
        return df[col_list].to_numpy(), df[target].to_numpy()

def predict_model(data):
    return model.predict(data)
        

@app.route('/', methods=['GET', 'POST'])
def choose_model():
    try:
        button = ChooseButton()

        if button.validate_on_submit():
            if button.rf_model.data:
                return redirect(url_for("params_model", model="RandomForest"))
            elif button.gb_model.data:
                return redirect(url_for("params_model", model="GradientBoosting"))
        return render_template("choose.html", form=button)
    except Exception as exc:
        app.logger.info('Exception in choose: {0}'.format(exc))
        return render_template("choose.html", form=button)


@app.route('/params', methods=['GET', 'POST'])
def params_model():
    try:
        model_type = request.args.get("model")
        if model_type == "RandomForest":
            params = Params_RF()
        elif model_type == "GradientBoosting":
            params = Params_GB()
        else:
            return redirect(url_for("choose_model"))
        
        default_button = Default_params()

        if request.method == 'POST' and default_button.validate_on_submit():
            default_button.data["button"] = False
            if model_type == "RandomForest":
                params.n_estimators.data = 25
                params.max_depth.data = default_param
                params.feature_subsample_size.data = default_param
            elif model_type == "GradientBoosting":
                params.n_estimators.data = 500
                params.max_depth.data = default_param
                params.feature_subsample_size.data = default_param
                params.learning_rate.data = 0.1

        if request.method == 'POST' and params.validate_on_submit():
            params_dict = {"n_estimators": params.n_estimators.data,
                           "max_depth": params.max_depth.data,
                           "feature_subsample_size": params.feature_subsample_size.data}
            if model_type == "GradientBoosting":
                params_dict["learning_rate"] = params.learning_rate.data
            session["params"] = params_dict
            session["model_type"] = model_type
            session["target"] = params.target.data
            return redirect(url_for("train_model"))

        return render_template("params.html", params=params, 
                    default_button=default_button, default_param=default_param)
    except Exception as exc:
        app.logger.info('Exception in params: {0}'.format(exc))
        return render_template("params.html", params=params)

error_list = []

class Error_pred:
    train = ""
    val = ""

@app.route('/params/train', methods=['GET', 'POST'])
def train_model():
    global error_list
    global model
    try:
        #buttons
        again_button = Again_button()
        predict_button = Submission_button()

        if request.method == "POST" and predict_button.validate_on_submit():
            predict_button.button.data = False
            with open(os.path.join(data_path, "submission.csv"), "w"):
                pass
            return redirect(url_for("submission"))
        if request.method == "POST" and again_button.validate_on_submit():
            again_button.button.data = False
            return redirect(url_for("choose_model"))
        #buttons

        params = session["params"]
        model_type = session["model_type"]
        target = session["target"]
        X_train, y_train = preprocess_data(train_df, target)
        if val_df is not None:
            X_val, y_val = preprocess_data(val_df, target)
        else:
            X_val, y_val = None, None

        #default keys check
        for key in list(params.keys()):
            if params[key] == default_param:
                if key == "n_estimators":
                    if model_type == "RandomForest":
                        params[key] = 25
                    else:
                        params[key] = 500
                else:
                    del params[key]
        #end check

        train_obj = training_model(model_type, params, X_train, y_train, X_val, y_val)
        for error in train_obj:
            err_obj = Error_pred()
            x, y = error
            x = "Train: {:.2f}".format(x**(1/2)) if x else ""
            y = "Validation: {:.2f}".format(y**(1/2)) if y else ""
            err_obj.train, err_obj.val = x, y
            error_list.append(err_obj)
        params["max_depth"] = model.max_depth
        params["feature_subsample_size"] = model.feature_subsample_size
        if model_type == "GradientBoosting":
            params["learning_rate"] = model.learning_rate

        return render_template("train.html", errors=error_list,
                        again=again_button, prediction=predict_button,
                        model_type=model_type, params=params)
    except Exception as exc:
        app.logger.info('Exception in train: {0}'.format(exc))
        model = None
        return redirect(url_for("choose_model"))


@app.route("/params/train/predict", methods=['GET', 'POST'])
def submission():
    try:
        target = session["target"]
        uploaded_file = CSV_upload()

        if request.method == "POST" and uploaded_file.validate_on_submit():
            uploaded_file.submit.data = False
            y_pred = predict_model(preprocess_data(val_df))
            y_pred = pd.DataFrame(y_pred, columns=[target])
            y_pred.to_csv(os.path.join(data_path, "submission.csv"), index=False)
            session["can_load"] = True

        return render_template("submission.html", upload=uploaded_file)
    except Exception as exc:
        app.logger.info('Exception in submission: {0}'.format(exc))
        return redirect(url_for("submission"))

@app.route("/uploads/submission")
def download_subm():
    try:
        if session.get("can_load"):
            return send_from_directory(directory=data_path, path="submission.csv")
        else:
            session["can_load"] = False
            return redirect(url_for("submission"))
    except Exception as exc:
        app.logger.info('Exception in download: {0}'.format(exc))
        return redirect(url_for("submission"))

@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for('choose_model'))
