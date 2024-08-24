from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import time


# экземпляр приложения
app = Flask(__name__)

# настройки для подключения к БД
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# инициализация БД
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    city = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'User({self.name},{self.age}, {self.city})'


class ModelResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(80), nullable=False)
    model_name = db.Column(db.String(80), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    execution_time = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'ModelResult({self.dataset_name}, {self.model_name}, {self.accuracy}, {self.execution_time})'


with app.app_context():
    db.create_all()


# маршрут для главной страницы
@app.route('/', methods=['GET', 'POST'])  # декоратор для связывания url (/) с функцией (home)
def home():
    global data
    greeting = ""
    plot_div = ""
    if request.method == 'POST':

        name = request.form.get('name')
        age = int(request.form.get('age'))
        city = request.form.get('city')

        greeting = f'Hello, {name} from {city}!'

        # добавление данных в БД
        new_user = User(name=name, age=age, city=city)
        db.session.add(new_user)
        db.session.commit()

        # извлечение данных
        users = User.query.all()
        data = [{'Name': user.name, 'Age': user.age, 'City': user.city} for user in users]

        # создание графика распределения возраста
        fig = px.histogram(data, x='Age', title='Age distribution', color_discrete_sequence=['#BD5353'])
        plot_div = pio.to_html(fig, full_html=False)

    return render_template('index.html', greeting=greeting, plot_div=plot_div)


# задание маршрута для страницы с датасетом ирисов
@app.route('/iris', methods=['GET', 'POST'])
def iris():
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target

    plot_div = ""
    accuracy_results = ""

    # построение графиков на основе данных
    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=iris_data.feature_names,
        color='species',
        category_orders={'species': ['setosa', 'versicolor', 'virginica']},
        color_continuous_scale=['#632616', '#c7a291', '#bf263a'],
        title='Iris Dataset Scatter Matrix',
        labels={col: col.replace(" (cm)", "") for col in iris_data.feature_names}
    )
    scatter_matrix.update_layout(
        height=500,
        width=1088,
        autosize=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    plot_div = pio.to_html(scatter_matrix, full_html=False)

    if request.method == 'POST':
        # разделение данных
        (X_train, X_test,
         y_train, y_test) = train_test_split(iris_data.data,
                                             iris_data.target, train_size=0.3, random_state=42)

        # стандартизация
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # обучение модели
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Support Vector Machines": SVC(kernel='rbf', gamma=0.1, C=1.0),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        }

        accuracy_results = {}
        with db.session.begin():
            for name, model in models.items():
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
                execution_time = round(time.time() - start_time, 4)
                accuracy_results[name] = accuracy
                model_result = ModelResult(dataset_name='Iris Dataset', model_name=name, accuracy=accuracy, execution_time=execution_time)
                db.session.add(model_result)

        # создание столбчатой диаграммы с результатами
        bar_chart = go.Figure([go.Bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()),
                                      marker=dict(color='#8a212e'))])
        bar_chart.update_layout(title="Model Accuracy Comparison", xaxis_title="Model", yaxis_title="Accuracy")
        plot_div += pio.to_html(bar_chart, full_html=False)

    return render_template('iris.html', plot_div=plot_div, accuracy_results=accuracy_results)


# задание маршрута для страницы с датасетом вин
@app.route('/wine', methods=['GET', 'POST'])
def wine():
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target

    table_html = df.to_html(index=False)
    plot_div_histogram = ""
    plot_div_comparison = ""
    accuracy_results = ""
    plot_div_dynamic = ""

    # построение распределения алкогольного содержания
    fig = px.histogram(df, x='alcohol', title='Distribution of Alcohol Content',
                       color_discrete_sequence=['#BD5353'])
    plot_div_histogram = pio.to_html(fig, full_html=False)

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'compare_models':
            # разделение данных
            (X_train, X_test,
             y_train, y_test) = train_test_split(wine_data.data,
                                                 wine_data.target, train_size=0.3, random_state=42)

            # стандартизация
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # обучение модели
            models = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

            accuracy_results = {}
            with db.session.begin():
                for name, model in models.items():
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
                    execution_time = round(time.time() - start_time, 4)
                    accuracy_results[name] = accuracy
                    model_result = ModelResult(dataset_name='Wine Dataset', model_name=name, accuracy=accuracy,
                                               execution_time=execution_time)
                    db.session.add(model_result)

            # создание столбчатой диаграммы с результатами
            bar_chart = go.Figure([go.Bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()),
                                          marker=dict(color='#8a212e'))])
            bar_chart.update_layout(title="Model Accuracy Comparison", xaxis_title="Model", yaxis_title="Accuracy")
            plot_div_comparison = pio.to_html(bar_chart, full_html=False)

        elif form_type == 'dynamic_plot':
            # динамический график
            x_axis = request.form.get('x_axis')
            y_axis = request.form.get('y_axis')
            plot_type = request.form.get('plot_type')

            if plot_type == 'scatter':
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {x_axis} vs {y_axis}',
                                 color_discrete_sequence=['#BD5353'])
            elif plot_type == 'histogram':
                fig = px.histogram(df, x=x_axis, title=f'Histogram of {x_axis}', color_discrete_sequence=['#BD5353'])
            elif plot_type == 'box':
                fig = px.box(df, x=x_axis, title=f'Box Plot of {x_axis}', color_discrete_sequence=['#BD5353'])

            plot_div_dynamic = pio.to_html(fig, full_html=False)

    return render_template('wine.html', table_html=table_html, plot_div_histogram=plot_div_histogram,
                           plot_div_comparison=plot_div_comparison, accuracy_results=accuracy_results,
                           plot_div_dynamic=plot_div_dynamic, feature_names=wine_data.feature_names)


# маршрут для страницы с результатами классификации
@app.route('/results', methods=['GET'])
def results():
    sort_by = request.args.get('sort_by', default='model_name')
    sort_order = request.args.get('sort_order', default='asc')

    if sort_by == 'model_name':
        if sort_order == 'asc':
            model_results = ModelResult.query.order_by(ModelResult.model_name.asc()).all()
        else:
            model_results = ModelResult.query.order_by(ModelResult.model_name.desc()).all()
    elif sort_by == 'accuracy':
        if sort_order == 'asc':
            model_results = ModelResult.query.order_by(ModelResult.accuracy.asc()).all()
        else:
            model_results = ModelResult.query.order_by(ModelResult.accuracy.desc()).all()

    return render_template('results.html', model_results=model_results, sort_by=sort_by, sort_order=sort_order)


if __name__ == '__main__':  # запуск осуществляется непосредственно из среды
    app.run(debug=True)
