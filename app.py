import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


processed_data = pd.read_csv("sample_processed_data.csv")
processed_data_X = processed_data.drop(["number_of_product_units", "generic_holiday"],axis=1).values
processed_data_Y=processed_data["number_of_product_units"].values
transformer = RobustScaler().fit(processed_data_X)
processed_data_X = transformer.transform(processed_data_X)
regressor_RF = RandomForestRegressor(n_estimators=13, random_state=101)
model_final = regressor_RF.fit(processed_data_X, processed_data_Y)
model = model_final


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def ValuePredictor(product_type, cost_per_unit, time_delivery, revenue, day_of_week):

    X = [[product_type, cost_per_unit, time_delivery, revenue, day_of_week]]

    X_scale = transformer.transform(X)
    prediction = model.predict(X_scale)

    return prediction


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_from_page = request.form.to_dict()
        print('one', data_from_page)

        def get_key_value(val):
            # page value loop
            for key, value in data_from_page.items():
                if val == key:
                    print('data from page:', value)

                    if val == 'product_type':
                        return int(value)

                    if val == 'cost_per_unit':
                        return int(value)

                    if val == 'revenue':
                        return int(value)

                    if val == 'day_of_week':
                        return int(value)

                    if val == 'time_delivery':
                        return int(value)

                    else:
                        print("Error: Doesn't match the keys with values")
                        return "Error: Doesn't match the keys with values"

            return "The key doesn't exist, check again"

        # function calls
        product_type = get_key_value('product_type')
        cost_per_unit = get_key_value('cost_per_unit')
        revenue = get_key_value('revenue')
        day_of_week = get_key_value('day_of_week')
        time_delivery = get_key_value('time_delivery')

        prediction = ValuePredictor(product_type, cost_per_unit, time_delivery, revenue, day_of_week)

        print('prediction  = ', prediction)

        return render_template('index.html', prediction_text= int(prediction))
 
 
if __name__ == "__main__":
    app.run(debug=True)
