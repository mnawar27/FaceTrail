# import sqlite3
# import csv

# db_name = "facetrail.db"

# conn = sqlite3.connect(db_name)
# cursor = conn.cursor()

# cursor.execute('''
# CREATE TABLE IF NOT EXISTS Persons (
#     id INTEGER PRIMARY KEY,
#     name TEXT NOT NULL,
#     dob DATE NOT NULL,
#     gender TEXT NOT NULL,
#     email TEXT NOT NULL,
#     phonenumber TEXT NOT NULL
# );
# ''')

# csv_file = "src/info.csv"
# with open(csv_file, "r") as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         cursor.execute('''
#         INSERT INTO Persons (id, name, dob, gender, email, phonenumber)
#         VALUES (?, ?, ?, ?, ?, ?);
#         ''', (row["ID"], row["Name"], row["DOB"], row["Gender"], row["Email"], row["Phonenumber"]))

# conn.commit()
# conn.close()

# print(f"Database '{db_name}' created and populated successfully!")


import mysql.connector
import csv

db_config = {
    'host': 'localhost',
    'user': 'mnawar', 
    'password': 'thisistheway', 
    'database': 'facetrail' 
}

conn = mysql.connector.connect(
    host=db_config['host'],
    user=db_config['user'],
    password=db_config['password']
)
cursor = conn.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS facetrail;")
conn.database = db_config['database']

cursor.execute('''
CREATE TABLE IF NOT EXISTS Persons (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    dob DATE NOT NULL,
    gender VARCHAR(10) NOT NULL,
    email VARCHAR(100) NOT NULL,
    phonenumber VARCHAR(20) NOT NULL
);
''')

csv_file = "src/info.csv"
with open(csv_file, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        cursor.execute('''
        INSERT INTO Persons (id, name, dob, gender, email, phonenumber)
        VALUES (%s, %s, %s, %s, %s, %s);
        ''', (row["ID"], row["Name"], row["DOB"], row["Gender"], row["Email"], row["Phonenumber"]))

conn.commit()
conn.close()

print("MySQL database 'facetrail' created and populated successfully!")
