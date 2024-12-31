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
        product_name varchar(200) NOT NULL,
        barcode varchar(100) NOT NULL UNIQUE,
        measurement varchar(100) NOT NULL, 
        cost_price float NOT NULL,
        selling_price float NOT NULL, 
        quantity integer NOT NULL
        )
        """

# create_all_record_table = """
#         CREATE TABLE all_record (
#         id integer PRIMARY KEY AUTO_INCREMENT,
#         year datetime NOT NULL DEFAULT TIMESTAMP,
#         product_name text NOT NULL,
#         barcode text NOT NULL UNIQUE,
#         measurement text NOT NULL, 
#         cost_price float NOT NULL,
#         selling_price float NOT NULL, 
#         quantity integer NOT NULL
#         )
#         """

# create_product_table = """
#         CREATE TABLE products (
#         productID integer PRIMARY KEY,
#         product_name text NOT NULL,
#         measurement text NOT NULL, 
#         date_added datetime default TIMESTAMPP,
#         barcode text NOT NULL UNIQUE
#         ) AUTO_INCREMENT = 1000
#         """

# create_sales_table = """
#                     CREATE TABLE sales (
#                     salesID integer PRIMARY KEY AUTO_INCREMENT,
#                     productID integer NOT NULL, 
#                     date_sold date DEFAULT TIMESTAMP,
#                     price float NOT NULL
#                     )
#                     """ 

# create_predict_table = """
#                         CREATE TABLE predicted (
#                         id integer PRIMARY KEY AUTO_INCREMENT,
#                         productID integer NOT NULL,
#                         product_name text NOT NULL,
#                         predicted_quantity integer NOT NULL
#                         )    
#                     """


# products_table = """
#             CREATE TABLE products (
#     productID INT AUTO_INCREMENT PRIMARY KEY,
#     product_name VARCHAR(255) NOT NULL,
#     barcode VARCHAR(100) UNIQUE NOT NULL,
#     date_added DATETIME DEFAULT CURRENT_TIMESTAMP
#     ) AUTO_INCREMENT = 1000

#     """


# records_table = """CREATE TABLE inventory (
#             inventoryID INT AUTO_INCREMENT PRIMARY KEY,
#             productID INT NOT NULL,
#             year DATE NOT NULL,
#             measurement VARCHAR(100) NOT NULL,
#             cost_price FLOAT NOT NULL,
#             selling_price FLOAT NOT NULL,
#             quantity INT NOT NULL,
#             FOREIGN KEY (productID) REFERENCES products(productID) ON DELETE CASCADE
#         )"""

sales_table = """
            CREATE TABLE sales (
            salesID VARCHAR(20) PRIMARY KEY, -- Alphanumeric ID
            date_sold DATETIME NOT NULL,
            total_amount FLOAT NOT NULL
        )
            """


sales_items_table = """
                    CREATE TABLE sales_items (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sales_id INT NOT NULL,
                    inventory_id INT NOT NULL,
                    quantity INT NOT NULL,
                    amount FLOAT NOT NULL,

                    FOREIGN KEY (sales_id) REFERENCES sales(salesID),
                    FOREIGN KEY (inventory_id) REFERENCES inventory(id)
                )
                """
items = """
            CREATE TABLE sales_items (
    id VARCHAR(20) PRIMARY KEY, -- Alphanumeric ID
    sales_id VARCHAR(20) NOT NULL,
    inventory_id INT NOT NULL,
    quantity INT NOT NULL,
    amount FLOAT NOT NULL,
    FOREIGN KEY (sales_id) REFERENCES sales(salesID) ON DELETE CASCADE,
    FOREIGN KEY (inventory_id) REFERENCES inventory(id) ON DELETE CASCADE
)"""

cursor.execute(items)
conn.close()
# salesID, PRODUCTID, product_name, qty_sold, total_amount 
# FOREIGN KEY (productID) REFERENCES inventory(id) ON DELETE SET NULL

# predictions_table = """CREATE TABLE predictions (
#                         predictionID INT AUTO_INCREMENT PRIMARY KEY,
#                         productID INT NOT NULL,
#                         predicted_quantity INT NOT NULL,
#                         prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
#                         FOREIGN KEY (productID) REFERENCES products(productID) ON DELETE CASCADE
#                     )
                    
#                     """


# query = [create_all_record_table, create_product_table, create_sales_table, create_predict_table]
# queries = [products_table, records_table, sales_table, predictions_table]
# tbls = [create_inventory_table, sales_table]
# for query in tbls:
#     cursor.execute(query)
# conn.close()

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

