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
        quantity integer NOT NULL,
        customer_rating float NOT NULL,
        season varchar(100) NOT NULL
        )
        """

create_sales_table = """
            CREATE TABLE sales (
            salesID VARCHAR(20) PRIMARY KEY, -- Alphanumeric ID
            date_sold DATETIME NOT NULL,
            total_amount FLOAT NOT NULL
        )
            """


create_sales_items_table = """
            CREATE TABLE sales_items (
            id VARCHAR(20) PRIMARY KEY, -- Alphanumeric ID
            sales_id VARCHAR(20) NOT NULL,
            inventory_id INT NOT NULL,
            quantity INT NOT NULL,
            amount FLOAT NOT NULL,
            FOREIGN KEY (sales_id) REFERENCES sales(salesID) ON DELETE CASCADE,
            FOREIGN KEY (inventory_id) REFERENCES inventory(id) ON DELETE CASCADE
        )"""
        
# alter and update customer rating column in inventory table
alter_inventory = """ALTER TABLE inventory ADD COLUMN customer_rating FLOAT"""

update_inventory = """UPDATE inventory 
                        SET customer_rating = ROUND(RAND() * 4 + 1, 1)
                        where id >= 1"""


queries = [create_inventory_table, create_sales_table, create_sales_items_table]


# for query in queries:
#     cursor.execute(query)


cursor.execute(create_inventory_table)

conn.close()