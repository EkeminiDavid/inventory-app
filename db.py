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

query = """
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

