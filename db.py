#本文档存储了连接数据库的信息

import pymysql
from pymysql.cursors import DictCursor
db_config = {
    'host': '81.68.103.209', # 主机
    'user': 'root',       #用户名
    'password': 'root',  #密码
    'port': 10086,         #端口 3306
    'database':'testuser'   #数据库名
}


def get_all(sql):
    print(sql)
    conn = pymysql.connect(**db_config)#数据库连接
    cursor = conn.cursor()
    cursor.execute(sql)
    info = cursor.fetchall()
    conn.close()
    return info
def findBy(sql,value):
    print(sql)
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(sql,value)
    info = cursor.fetchall()
    conn.close()
    return info
def insert(sql,value):#数据库信息插入
    print(sql)
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    try:
        # Execute the SQL command
        cursor.execute(sql,value)
        # Commit your changes in the database
        conn.commit()
        conn.close()
    except:
        # Rollback in case there is any error
        conn.rollback()
        conn.close()
        return False

    return True
