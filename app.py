from flask import Flask, request, jsonify
# import sqlite3
import pymysql
import datetime

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine


year = datetime.datetime.now().date()
year = year.strftime("%Y-%m-%d")
# print("Current Date:", current_date)


app = Flask(__name__)

# Create database connection 
def db_connect():
  conn = None
  try:
      conn = pymysql.connect(
      host="sql7.freesqldatabase.com",
      database="sql7751294",
      user="sql7751294",
      password="IWWaeXWra2",
      charset="utf8mb4",
      cursorclass=pymysql.cursors.DictCursor,
      port=3306
        )
      
  except pymysql.Error as e:
      print(e)
  return conn


@app.route('/inventory', methods=['GET', 'POST'])
def get_inventory():
    conn = db_connect()
    cursor = conn.cursor()

    data = request.get_json()

    if request.method == 'GET':
        cursor.execute("SELECT * FROM inventory")
        inventory = [
            dict(id=row['id'], year=row['year'], product_name=row['product_name'], barcode=row['barcode'],\
                 measurement=row['measurement'], cost_price=row['cost_price'], selling_price=row['selling_price'],\
                    quantity=row['quantity'])
                 for row in cursor.fetchall()
        ]
        if inventory is not None:
            return jsonify(inventory)

      
    if request.method == 'POST':
        print("get post req")
        new_year = year #request.form['year']
        new_product = data['product_name']
        new_code = data['barcode']
        new_measurement = data['measurement']
        new_cost_price = data['cost_price']
        new_selling_price = data['selling_price']
        new_qty = data['quantity']
        # new_product = request.form['product_name']
        # new_code = request.form['barcode']
        # new_measurement = request.form['measurement']
        # new_cost_price = request.form['cost_price']
        # new_selling_price = request.form['selling_price']
        # new_qty = request.form['quantity']

        sql = """INSERT INTO inventory(year, product_name, barcode, measurement, cost_price, selling_price, quantity) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
        cursor.execute(sql, (new_year, new_product, new_code, new_measurement, new_cost_price, new_selling_price, new_qty))
        conn.commit()
        return f"Products with the id: {cursor.lastrowid} created succesfully", 201
        

@app.route('/inventory/<int:id>', methods=['GET', 'PUT', 'DELETE'])
def single_inv(id):
    conn = db_connect()
    cursor = conn.cursor()
    data = request.get_json()
    
    inventory = None

    if request.method == 'GET':
        cursor.execute("SELECT * FROM inventory WHERE id=%s", (id,))
        rows = cursor.fetchall()
        for r in rows:
            inventory = r
            if inventory is not None:
                return jsonify(inventory), 200
            else:
                return "Something wrong", 404
            
    if request.method == 'PUT':
        sql = """
              UPDATE inventory
              SET year = %s,
              product_name = %s,
              barcode=%s,
              measurement=%s,
              cost_price=%s,
              selling_price=%s,
              quantity=%s
              WHERE id=%s
              """
        # year = year #request.form['year']
        # product_name = request.form['product_name']
        # barcode = request.form['barcode']
        # measurement = request.form['measurement']
        # cost_price = request.form['cost_price']
        # selling_price = request.form['selling_price']
        # quantity = request.form['quantity']

        product_name = data['product_name']
        barcode = data['barcode']
        measurement = data['measurement']
        cost_price = data['cost_price']
        selling_price = data['selling_price']
        quantity = data['quantity']


        updated_inv = {
            'id': id,
            'year': year,
            'product_name': product_name,
            'barcode': barcode,
            'measurement': measurement,
            'cost_price': cost_price,
            'selling_price': selling_price,
            'quantity': quantity
        }
        cursor.execute(sql, (year,product_name, barcode, measurement, cost_price, selling_price, quantity, id))
        conn.commit()

        return jsonify(updated_inv)
    # Delete item
    if request.method == 'DELETE':
        sql = """ DELETE FROM inventory WHERE id=%s"""
        cursor.execute(sql, (id,))
        conn.commit()
        return "The product with id: {} has been deleted.".format(id), 200




# Fetch data from the database
def load_data():
    
    host="sql7.freesqldatabase.com"
    database="sql7751294"
    username="sql7751294"
    password="IWWaeXWra2"
    charset="utf8mb4"
    cursorclass=pymysql.cursors.DictCursor
    port=3306
    
    connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'

# Create an engine using SQLAlchemy
    engine = create_engine(connection_string)
    query = 'SELECT product_name, year, quantity FROM inventory'
    # cursor.execute(query)
    
    df = pd.read_sql(query, engine)
    # conn.close()
    return df

# Load data from the SQL database
data = load_data()

#lower all product name
data['product_name'] = data['product_name'].apply(lambda x: x.lower())
# Handle missing values (imputation or removal)
data['month'] = pd.to_datetime(data['year'])
data['month_num'] = data['month'].dt.month


# Encode product names to numerical values
le = LabelEncoder()
data['product_name_encoded'] = le.fit_transform(data['product_name'])

import pickle
# with open('label_encoder.pkl', 'wb') as f:
#     pickle.dump(le, f)

# Impute missing values in 'month_num' and 'product_name_encoded' columns
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median' or 'most_frequent'
data[['month_num', 'product_name_encoded']] = imputer.fit_transform(data[['month_num', 'product_name_encoded']])

# Features and target variable
X = data[['product_name_encoded', 'month_num']]
y = data['quantity']
# print(y.shape)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)


#Predict product quantity
def predict_quantity(product_name, month_num):
    current_classes = list(le.classes_)
    

    if product_name.lower() not in current_classes:
        # print(le.classes_)
        current_classes.append(product_name)
        le.classes_ = np.array(current_classes)


 
   # product_name_encoded = le.transform([product_name])[0]
    product_name_encoded = le.transform([product_name])
            
    input_data = pd.DataFrame([[product_name_encoded, month_num]], columns=['product_name_encoded', 'month_num'])
    prediction = model.predict(input_data)
    return max(0, int(prediction[0])) # Ensure non-negative prediction

@app.route('/predict_product_quantity', methods=['POST'])
def predict_product_quantity():
    # if request.method == 'POST':
    postedData = request.json
    # data =request.get_json(force=True)

    if not postedData:
        return jsonify({'error': 'Invalid or missing JSON data. Ensure Content-Type is application/json.'}), 400

    try:
        product_name = postedData['product_name']
        month_num = postedData['month_num']
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400

    try:
        
        # Call the prediction function with encoded product name
        predicted_quantity = predict_quantity(product_name, month_num)
        return jsonify({'product_name': product_name, 'predicted_quantity': predicted_quantity})

    except Exception as e:

        return jsonify({'error': str(e)}), 500

# id,name,category,quantity,price,supplier
if __name__ == '__main__':
    app.run(debug=True)