import sqlite3
import pymysql
import pymysql.cursors


conn = pymysql.connect(
    host="sql7.freesqldatabase.com",
    database="sql7751294",
    user="sql7751294",
    password="IWWaeXWra2",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)

cursor = conn.cursor()

create_inventory_table = """
        CREATE TABLE inventory (
        id integer PRIMARY KEY AUTO_INCREMENT,
        year date NOT NULL,
        product_name text NOT NULL,
        barcode text NOT NULL,
        measurement text NOT NULL, 
        cost_price float NOT NULL,
        selling_price float NOT NULL, 
        quantity integer NOT NULL
        )
        """


create_product_table = """
        CREATE TABLE products (
        productID integer PRIMARY KEY,
        product_name text NOT NULL,
        date_added datetime default TIMESTAMPP,
        barcode text NOT NULL
        ) AUTO_INCREMENT = 1000
        """

create_sales_table = """
                    CREATE TABLE sales (
                    salesID integer PRIMARY KEY AUTO_INCREMENT,
                    productID integer NOT NULL, 
                    date_sold date DEFAULT TIMESTAMP,
                    price float NOT NULL
                    )
                    """ 

create_predict_table = """
                        CREATE TABLE predicted (
                        id integer PRIMARY KEY AUTO_INCREMENT,
                        productID integer NOT NULL,
                        product_name text NOT NULL,
                        predicted_quantity integer NOT NULL
                        )    
                    """
query = [create_inventory_table, create_product_table, create_sales_table, create_predict_table]
for query in queries:
    
    cursor.execute(query)
    conn.close()

# conn = sqlite3.connect('inventory.sqlite')

# cursor = conn.cursor()

# query = """CREATE TABLE inventory (
#             id integer PRIMARY KEY, 
#             name text NOT NULL,
#             category text NOt NULL,
#             price integer NOT NULL,
#             quantity integer NOT NULL, 
#             supplier integer NOT NULL
# )"""

# cursor.execute(query)

