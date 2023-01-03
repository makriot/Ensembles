import os
import base64
from io import BytesIO
from math import isnan

from flask import Flask, render_template
from flask import request, url_for, redirect
from flask import session, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired

from wtforms.validators import DataRequired, ValidationError
from wtforms.validators import StopValidation
from wtforms import FileField, SubmitField, StringField
from wtforms import IntegerField, FloatField

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ensembles_web as ens

matplotlib.use('Agg')

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'snowden'
Bootstrap(app)

DATA_PATH = "./data"

model = None
train_df = None
val_df = None
default_param = -1
frequency_loss = 10


## checkers
class NumberChecker():

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


class TrainCsvDataChecker():

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


class TestCsvDataChecker():

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


class TargetChecker():

    def __call__(self, form, field):
        if train_df is None:
            raise ValidationError("No valid dataset is provided")
        if field.data not in train_df.columns:
            raise ValidationError(field.gettext("No such column: '%s'.") % field.data)


class FileOptional():

    def __call__(self, form, field):
        if not field.data:
            field.errors[:] = []
            raise StopValidation()


## fields
class ChooseButton(FlaskForm):
    rf_model = SubmitField("Random Forest")
    gb_model = SubmitField("Gradient Boosting")


class ParamsRF(FlaskForm):
    n_estimators = IntegerField("Number of trees",
                                validators=[DataRequired(), NumberChecker(min=1)])
    max_depth = IntegerField("Maximum depth",
                             validators=[DataRequired(), NumberChecker(min=1)])
    feature_subsample_size = IntegerField("Number of features",
                                          validators=[DataRequired(), NumberChecker(min=1)])
    train_file = FileField('Train dataset', validators=[
        FileRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!'),
        TrainCsvDataChecker()
    ])
    target = StringField('Target label',
                         validators=[DataRequired(), TargetChecker()])
    test_file = FileField('Validation dataset (optional)', validators=[
        FileOptional(),
        FileAllowed(['csv'], 'CSV only!'),
        TestCsvDataChecker()
    ])
    button = SubmitField("Train")


class ParamsGB(FlaskForm):
    n_estimators = IntegerField("Number of trees",
                                validators=[DataRequired(), NumberChecker(min=1)])
    learning_rate = FloatField("Learning rate",
                               validators=[DataRequired(), NumberChecker(min=0, greater=True)])
    max_depth = IntegerField("Maximum depth",
                             validators=[DataRequired(), NumberChecker(min=1)])
    feature_subsample_size = IntegerField("Number of features",
                                          validators=[DataRequired(), NumberChecker(min=1)])
    train_file = FileField('Train dataset', validators=[
        FileRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!'),
        TrainCsvDataChecker()
    ])
    target = StringField('Target label',
                         validators=[DataRequired(), TargetChecker()])
    test_file = FileField('Validation dataset (optional)', validators=[
        FileOptional(),
        FileAllowed(['csv'], 'CSV only!'),
        TestCsvDataChecker()
    ])
    button = SubmitField("Train")


class DefaultParams(FlaskForm):
    but_def = SubmitField("Fill default parameters")


class AgainButton(FlaskForm):
    button1 = SubmitField("Try again")


class SubmissionButton(FlaskForm):
    button2 = SubmitField("Predict")


class CSVUpload(FlaskForm):
    file_obj = FileField('Dataset', validators=[
        FileRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!'),
        TestCsvDataChecker()
    ])
    submit = SubmitField("Submit")


def create_plot(errors_train, errors_val):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(errors_train) + 1), errors_train, label="train")
    if len(errors_val) and errors_val[0]:
        ax.plot(range(1, len(errors_train) + 1), errors_val, label="validation")
    ax.grid(True, alpha=0.5)
    ax.legend()
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Iterations")
    return fig


def plot_png(errors_train, errors_val):
    fig = create_plot(errors_train, errors_val)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"


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

        if request.method == "POST" and button.validate_on_submit():
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
        default_button = DefaultParams()
        model_type = request.args.get("model")

        if request.method == 'POST' and default_button.but_def and default_button.validate():
            if model_type == "RandomForest":
                params = ParamsRF(n_estimators=25, max_depth=default_param,
                                  feature_subsample_size=default_param)
            elif model_type == "GradientBoosting":
                params = ParamsGB(n_estimators=500, max_depth=default_param,
                                  feature_subsample_size=default_param, learning_rate=0.1)
        else:
            if model_type == "RandomForest":
                params = ParamsRF()
            elif model_type == "GradientBoosting":
                params = ParamsGB()
            else:
                return redirect(url_for("choose_model"))

        if request.method == 'POST' and params.validate_on_submit() and params.button.data:
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
                               default_button=default_button,
                               default_param=default_param)
    except Exception as exc:
        app.logger.info('Exception in params: {0}'.format(exc))
        return render_template("params.html", params=params)


class Error_pred:
    iteration = ""
    train = ""
    val = ""


@app.route('/params/train', methods=['GET', 'POST'])
def train_model():
    global model
    error_list = []
    errors_train = []
    errors_val = []
    try:
        # buttons
        again_button = AgainButton()
        predict_button = SubmissionButton()

        if request.method == 'POST' and again_button.button1.data and again_button.validate():
            return redirect(url_for("choose_model"))

        if request.method == 'POST' and predict_button.button2.data and predict_button.validate():
            with open(os.path.join(DATA_PATH, "submission.csv"), "w"):
                pass
            return redirect(url_for("submission"))
        # buttons

        params = session["params"]
        model_type = session["model_type"]
        target = session["target"]
        X_train, y_train = preprocess_data(train_df, target)
        if val_df is not None:
            X_val, y_val = preprocess_data(val_df, target)
        else:
            X_val, y_val = None, None

        # default keys check
        for key in list(params.keys()):
            if params[key] == default_param:
                if key == "n_estimators":
                    if model_type == "RandomForest":
                        params[key] = 25
                    else:
                        params[key] = 500
                else:
                    del params[key]
        # end check

        train_obj = training_model(model_type, params, X_train, y_train, X_val, y_val)
        for i, error in enumerate(train_obj):
            err_obj = Error_pred()
            x, y = error
            errors_train.append(x)
            errors_val.append(y)
            iteration = "Iteration: {}".format(i + 1) if x else ""
            x = "Train: {:.2f}".format(x**(1/2)) if x else ""
            y = "Validation: {:.2f}".format(y**(1/2)) if y else ""
            err_obj.iteration, err_obj.train, err_obj.val = iteration, x, y
            error_list.append(err_obj)
        errors_frequency = []
        step = max(1, params["n_estimators"]//frequency_loss)
        for i in range(frequency_loss):
            errors_frequency.append(error_list[i * step])
        final_err = error_list[-1]
        final_err.iteration = f"Final {final_err.iteration}"
        errors_frequency.append(final_err)
        path_graphic = plot_png(errors_train, errors_val)
        params["max_depth"] = model.max_depth
        params["feature_subsample_size"] = model.feature_subsample_size
        if model_type == "GradientBoosting":
            params["learning_rate"] = model.learning_rate

        return render_template("train.html", errors=errors_frequency,
                               again=again_button, prediction=predict_button,
                               model_type=model_type, params=params,
                               graphic_source=path_graphic)
    except Exception as exc:
        app.logger.info('Exception in train: {0}'.format(exc))
        model = None
        return redirect(url_for("choose_model"))


@app.route("/params/train/predict", methods=['GET', 'POST'])
def submission():
    try:
        if model is None:
            redirect(url_for("choose_model"))
        target = session["target"]
        uploaded_file = CSVUpload()

        if request.method == "POST" and uploaded_file.validate_on_submit():
            uploaded_file.submit.data = False
            y_pred = predict_model(preprocess_data(val_df))
            y_pred = pd.DataFrame(y_pred, columns=[target])
            y_pred.to_csv(os.path.join(DATA_PATH, "submission.csv"), index=False)
            session["can_load"] = True

        return render_template("submission.html", upload=uploaded_file)
    except Exception as exc:
        app.logger.info('Exception in submission: {0}'.format(exc))
        return redirect(url_for("submission"))


@app.route("/uploads/submission")
def download_subm():
    try:
        if session.get("can_load"):
            return send_from_directory(directory=DATA_PATH, path="submission.csv")
        else:
            session["can_load"] = False
            return redirect(url_for("submission"))
    except Exception as exc:
        app.logger.info('Exception in download: {0}'.format(exc))
        return redirect(url_for("submission"))


@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for('choose_model'))
