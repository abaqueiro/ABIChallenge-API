#!/usr/bin/python3
# how to read data from postgresql into a pandas.DataFrame
import psycopg2 as pgsql
import pandas

cnn = pgsql.connect( host="127.0.0.1", database="pru", user="postgres", password="some_G00D-PwD" )
cursor = cnn.cursor()
cursor.execute("select emp_id, emp_name from employee")
qr = cursor.fetchall()
#cnn.commit()
cursor.close()

dat = pandas.DataFrame(qr,columns=["id","name"])
dat.info()
print("DATA:")
print(dat)
 