from sys import argv
import pandas.io.sql as psql
import psycopg2                                         #for postgresql
"""import cx_Oracle                                     #for Oracle
    import sqlite3                                      #for sqlite
#might need install and change code structure."""
from datetime import datetime as dt

conn = psycopg2.connect("dbname='storemanagement' user='postgres' host='localhost' password='admin'")
if __name__ == "__main__":
    if (len(argv)) != 2:
        print("Wrong number of arguments. Check again!")
    else:
        print("Remember that input has to be inside double quote and follow the format dd/mm/yyyy.")
        print("Otherwise it will not work properly.")
        date = dt.strptime(argv[1], "%d/%m/%Y")
        command = "select * from storeinfo.store where timesold = \'" + date.strftime("%Y-%m-%d") + "\'"
        df = psql.read_sql(command, conn)
        df.to_csv(path_or_buf = "~/VSCode/Python/machinelearning/exercise/test.csv", index = False)
        print("Done! Check file test.csv in ~/VSCode/Python/machinelearning/exercise to see the result")