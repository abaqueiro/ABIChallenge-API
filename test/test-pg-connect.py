#!/usr/bin/python3
import psycopg2 as dbo

cnn = dbo.connect( host="127.0.0.1", database="pru", user="postgres", password="some_G00D-PwD" )
cursor = cnn.cursor()

cursor.execute(
    "select emp_id, emp_name from employee where emp_id=%(emp_id)s"
    , {"emp_id": "1" }
)
r = cursor.fetchone()
someone_name = r[1]
print(someone_name)

cursor.execute(
    "insert into employee_of_month (emp_name) values (%(emp_name)s)"
    , {"emp_name": someone_name}
)
cnn.commit()

cursor.close()
