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

    if request.method == 'GET':
        cursor.execute("SELECT * FROM inventory")
        inventory = [
            dict(id=row['id'], year=row['year'], product_name=row['product_name'], barcode=row['barcode'],\
                 measurement=row['measurement'], cost_price=row['cost_price'], selling_price=row['selling_price'],\
                    quantity=row['quantity'])
                 for row in cursor.fetchall()
        ]
        if inventory is not None:
            returnMessage = {
                'message': "Inventory List",
                'status_code': 200,
                'body': inventory
            }
            return jsonify(returnMessage)

    postData = request.get_json()
    if request.method == 'POST':
        print("get post req")
        new_year = year 
        new_product = postData['product_name']
        new_code = postData['barcode']
        new_measurement = postData['measurement']
        new_cost_price = postData['cost_price']
        new_selling_price = postData['selling_price']
        new_qty = postData['quantity']

        sql = """INSERT INTO inventory(year, product_name, barcode, measurement, cost_price, selling_price, quantity) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
        cursor.execute(sql, (new_year, new_product, new_code, new_measurement, new_cost_price, new_selling_price, new_qty))
        conn.commit()

        returnMessage = {
            'message': f"Products with the id: {cursor.lastrowid} created succesfully",
            "status_code": 201,
            "body": {
                "year": new_year,
                "barcode": new_code,
                "cost_price": new_cost_price,
                "measurement": new_measurement,
                "product_name": new_product,
                "quantity": new_qty,
                "selling_price": new_selling_price
            }
        }

        return jsonify(returnMessage)
        # return f"Products with the id: {cursor.lastrowid} created succesfully", 201
        

@app.route('/inventory/<int:id>', methods=['GET', 'PUT', 'DELETE'])
def single_inv(id):
    conn = db_connect()
    cursor = conn.cursor()
    inventory = None

    if request.method == 'GET':
        cursor.execute("SELECT * FROM inventory WHERE id=%s", (id,))
        rows = cursor.fetchall()
        for r in rows:
            inventory = r
            if inventory is not None:
                returnMessage = {
                    'message': "Inventory Item {}".format(id),
                    'status_code': 200,
                    'body': inventory
                }
                return jsonify(returnMessage)
            else:
                returnMessage = {
                    'message': "error: ",
                    'status_code': 404,
                    'body': "Something wrong"
                }
                return jsonify(returnMessage)

    postData = request.get_json()    
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

        product_name = postData['product_name']
        barcode = postData['barcode']
        measurement = postData['measurement']
        cost_price = postData['cost_price']
        selling_price = postData['selling_price']
        quantity = postData['quantity']


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
        returnMessage = {
            'message': f"Updated item: {id}",
            'status_code': 200,
            'body': updated_inv
        }
        return jsonify(returnMessage)
    
    # Delete item
    postData = request.get_json()
    if request.method == 'DELETE':
        sql = """ DELETE FROM inventory WHERE id=%s"""
        cursor.execute(sql, (id,))
        conn.commit()
        returnMessage = {
            'message': "Product deleted",
            'status_code': 200,
            'body': "The product with id: {} has been deleted.".format(id)
        }
        return jsonify(returnMessage)




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

@app.route('/predict_quantity', methods=['POST'])
def predict_product_quantity():
    # if request.method == 'POST':
    data = request.json
    # data =request.get_json(force=True)

    if not data:
        returnMessage = {
            'message': 'data not found',
            'status_code': 400,
            'body': {'error': 'Invalid or missing JSON data. Ensure Content-Type is application/json.'}
        }
        return jsonify(returnMessage)

    try:
        product_name = data['product_name']
        month_num = data['month_num']
        current_classes = list(le.classes_)
    

        if product_name.lower() not in current_classes:
            # print(le.classes_)
            current_classes.append(product_name)
            le.classes_ = np.array(current_classes)
    except KeyError as e:
        returnMessage = {
            'message': 'error: ',
            'status_code': 400,
            'body': f'Missing key: {str(e)}'
        }
        return jsonify(returnMessage)

    try:
        
        # Call the prediction function with encoded product name
        predicted_quantity = predict_quantity(product_name, month_num)
        returnMessage = {
            'message': f'Predicted quantity for month {month_num}',
            'status_code': 500,
            'body': {'product_name': product_name, 'predicted_quantity': predicted_quantity}
        }
        return jsonify(returnMessage)

    except Exception as e:
        returnMessage = {
            'message': 'error: ',
            'status_code': 500,
            'body': f'Missing key: {str(e)}'
        }
        return jsonify(returnMessage)

# id,name,category,quantity,price,supplier
if __name__ == '__main__':
    app.run(debug=True)