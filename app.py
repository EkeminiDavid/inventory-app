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

@app.route('/total_product', methods=['GET'])
def total_product():
    conn = db_connect()
    cursor = conn.cursor()

    # if request.method == 'GET':
    try:
        cursor.execute("SELECT COUNT(id) AS total_product FROM inventory")
        count = cursor.fetchone()
        returnMessage = {
                'message': "Total product",
                'status_code': 200,
                'body': count
            }
        return jsonify(count)

    except Exception as e:
        returnMessage = {
                'message': "error",
                'status_code': 500,
                'body': {"error": str(e)}
            }
        return jsonify(returnMessage)

    finally:
        cursor.close()
        conn.close()



@app.route('/inventory', methods=['GET', 'POST'])
def get_inventory():
    conn = db_connect()
    cursor = conn.cursor()

    if request.method == 'GET':
        # cursor.execute("SELECT * FROM inventory")
        cursor.execute(""" SELECT id, year, product_name, \
                       barcode, measurement, cost_price, \
                       selling_price, SUM(quantity) AS quantity\
                        FROM inventory
                       GROUP BY product_name
                       """)
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
        
@app.route('/make_sales', methods=['GET', 'POST'])
def make_sales():
    conn = db_connect()
    cursor = conn.cursor()
    sales = None

    if request.method == 'GET':
        cursor.execute("SELECT * FROM sales")
        # rows = cursor.fetchall()

        sales = [
            dict(salesID=row['salesID'], inventoryID=row['inventoryID'], product_name=row['product_name'], date_sold=row['date_sold'], \
                 amount=row['amount'], measurement=row['measurement'], quantity_sold=row['quantity_sold'])
                    for row in cursor.fetchall()
        ]
        
        if sales is not None:
            returnMessage = {
                'message': "Sales history",
                'status_code': 200,
                'body': sales
            }
            return jsonify(returnMessage)
        

    postData = request.get_json()
    if request.method == 'POST':
        print("get post req")
        new_date_sold = year 
        new_name = postData['product_name']
        new_inventoryID = postData['inventoryID']
        new_measurement = postData['measurement']
        new_amount = postData['amount']
        new_qty = postData['quantity_sold']

        sql = """INSERT INTO sales(date_sold, product_name, inventoryID, measurement, amount, quantity_sold) VALUES (%s,%s,%s,%s,%s,%s)"""
        cursor.execute(sql, (new_date_sold, new_name, new_inventoryID, new_measurement, new_amount,  new_qty))
        conn.commit()

        returnMessage = {
            'message': f"Products with the id: {cursor.lastrowid} created succesfully",
            "status_code": 200,
            "body": {
                "year": new_date_sold,
                "Amount": new_amount,
                "measurement": new_measurement,
                "product_name": new_name,
                "quantity_sold": new_qty,
                'inventoryID': new_inventoryID
            }
        }

        return jsonify(returnMessage)            



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
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def get_week_number(date):
    """Get the week number (1-52/53) for a given date"""
    return int(date.strftime('%V'))

def predict_quantity(product_name, start_date, end_date):
    current_classes = list(le.classes_)
    
    if product_name.lower() not in current_classes:
        current_classes.append(product_name.lower())
        le.classes_ = np.array(current_classes)
    
    product_name_encoded = le.transform([product_name.lower()])
    predictions = {}
    
    # Convert start_date and end_date to datetime objects if they're strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate dates for each day in the range
    current_date = start_date
    while current_date <= end_date:
        month_num = current_date.month
        week_num = get_week_number(current_date)
        
        # Create a unique key for the week
        date_key = f"{current_date.year}-W{week_num:02d}"
        
        if date_key not in predictions:
            # Make prediction for this month (you might want to adjust the model
            # to take week numbers into account for more precise predictions)
            input_data = pd.DataFrame([[product_name_encoded[0], month_num]], 
                                    columns=['product_name_encoded', 'month_num'])
            prediction = model.predict(input_data)
            
            # For weekly predictions, divide monthly prediction by ~4.345 weeks per month
            weekly_prediction = max(0, int(prediction[0] / 4.345))
            predictions[date_key] = weekly_prediction
        
        current_date += timedelta(days=7)  # Move to next week
        
    return predictions

@app.route('/predict_quantity', methods=['POST'])
def predict_product_quantity():
    month_name = {1: 'January', 2:"February", 3:'March', 4:'April', 5:'May', 6:'June', 
                 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
    
    data = request.json

    if not data:
        returnMessage = {
            'message': 'data not found',
            'status_code': 400,
            'body': {'error': 'Invalid or missing JSON data. Ensure Content-Type is application/json.'}
        }
        return jsonify(returnMessage)
 
    try:
        # Get required parameters
        product_name = data['product_name']
        start_date = data['start_date']  # Format: YYYY-MM-DD
        end_date = data['end_date']      # Format: YYYY-MM-DD
        
        # Validate dates
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start > end:
                raise ValueError("Start date must be before end date")
                
            # Optional: Add validation for maximum date range
            max_days = 365  # Example: limit to 1 year
            if (end - start).days > max_days:
                raise ValueError(f"Date range cannot exceed {max_days} days")
                
        except ValueError as e:
            returnMessage = {
                'message': 'error: invalid date format or range',
                'status_code': 400,
                'body': str(e)
            }
            return jsonify(returnMessage)
            
    except KeyError as e:
        returnMessage = {
            'message': 'error: ',
            'status_code': 400,
            'body': f'Missing key: {str(e)}'
        }
        return jsonify(returnMessage)

    try:
        # Get predictions for the date range
        predictions = predict_quantity(product_name.lower(), start_date, end_date)
        
        # Format the response
        formatted_response = {
            'predictions': predictions,
            'metadata': {
                'product': product_name,
                'start_date': start_date,
                'end_date': end_date,
                'total_weeks': len(predictions),
                'total_predicted_quantity': sum(predictions.values())
            }
        }
        
        returnMessage = {
            'message': 'Weekly predictions generated successfully',
            'status_code': 200,
            'body': formatted_response
        }
        return jsonify(returnMessage)

    except Exception as e:
        returnMessage = {
            'message': 'error: ',
            'status_code': 500,
            'body': f'Prediction error: {str(e)}'
        }
        return jsonify(returnMessage)

if __name__ == '__main__':
    app.run(debug=True)