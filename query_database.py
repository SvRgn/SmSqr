from db.db_factory import DatabaseFactory
from db.schema import MessagesTable
from sqlalchemy import and_
import timestring
import os

#
# Adapt here port, password, user and db_name to the database in use
#

def requery_database_by_user(user_id):
    global DB
    DB = DatabaseFactory().create(port=3306, password="", user="", db_name="square")
    print("INFO: Collecting social media messages from db by USER")
    # SQL Statement: select messages.time, messages.user_id, messages.text, messages.service from messages where messages.user_id = '11201847'
    result = DB.session.query(MessagesTable.time, MessagesTable.user_id, MessagesTable.text,
                            MessagesTable.service).filter(MessagesTable.user_id == user_id)
    return result

def requery_database_by_user_and_time(user_id, time, timedelta):
    global DB
    DB = DatabaseFactory().create(port=3306, password="", user="", db_name="square")
    print("INFO: Collecting social media messages from db by USER in time range")
    tm = timestring.Date(time).date
    since = tm - timedelta(days=timedelta)
    until = tm + timedelta(days=timedelta)
    # SQL Statement: select messages.time, messages.user_id, messages.text, messages.service from messages where messages.user_id = '11201847' AND messages.time > '2018-04-19 23:59:59' AND messages.time  < '2018-04-21 23:59:59'
    result = DB.session.query(MessagesTable.time, MessagesTable.user_id, MessagesTable.text,
                            MessagesTable.service).filter(and_(MessagesTable.user_id == user_id, MessagesTable.time > str(since), MessagesTable.time < str(until)))
    return result


def requery_database_by_topword_and_time(topword, time, timedelta):
    global DB
    DB = DatabaseFactory().create(port=3306, password="", user="", db_name="square")
    print("INFO: Collecting social media messages from db by topword in time range")
    tm = timestring.Date(time).date
    since = tm - timedelta(days=timedelta)
    until = tm + timedelta(days=timedelta)
    # SQL Statement: select messages.time, messages.user_id, messages.text, messages.service from messages where messages.user_id = '11201847' AND messages.time > '2018-04-19 23:59:59' AND messages.time  < '2018-04-21 23:59:59'
    result = DB.session.query(MessagesTable.time, MessagesTable.user_id, MessagesTable.text,
                            MessagesTable.service).filter(and_(MessagesTable.text.contains(topword), MessagesTable.time > str(since), MessagesTable.time < str(until)))
    return result



def query_db_by_timespan_with_filewrite(since, until, to_file):
    global DB
    DB = DatabaseFactory().create(port=3306, password="", user="", db_name="square")
    print("INFO: Collecting social media messages from db")
    # SQL Statement: select messages.time, messages.user_id, messages.text, messages.service from messages where messages.time  > '2018-04-13 00:00:00' AND messages.time  < '2018-04-20 23:59:59' AND messages.text NOT LIKE '' AND messages.service NOT LIKE 'googlealerts' AND messages.service NOT LIKE 'rss' AND messages.service NOT LIKE 'facebook'
    data = DB.session.query(MessagesTable.time, MessagesTable.user_id, MessagesTable.text,
                            MessagesTable.service).filter(
        and_(MessagesTable.time > since, MessagesTable.time < until,
             MessagesTable.text != '', MessagesTable.service != 'googlealerts', MessagesTable.service != 'rss',
             MessagesTable.service != 'facebook'))
    # data = DB.session.query(MessagesTable.time, MessagesTable.user_id, MessagesTable.text,MessagesTable.service).filter(and_(MessagesTable.time > '2018-04-21 00:00:00', MessagesTable.time < '2018-04-28 23:59:59',MessagesTable.text != '', MessagesTable.service != 'googlealerts', MessagesTable.service != 'rss',MessagesTable.service != 'facebook'))

    print('INFO: Found ', data.count(), 'social media messages in db')
    print("INFO: Creating csv")
    if os.path.isfile(to_file):
        os.remove(to_file)
    db_data = open(to_file, 'a', encoding='utf8')
    db_data_result = []
    for message in data:
        db_data.write(
            str(message[0]) + '; ' + str(message[1]) + '; ' + 'hamburg' + ' ; ;"' + str(message[2]).rstrip("\n").replace(";", ":") + '"; ' + str(
                message[3]) + '\n')
        db_data_result.append(str(message[0]) + ';' + str(message[1]) + ';' + 'hamburg' + ';"' + str(message[2]).rstrip("\n").replace(";", ":") + '";' + str(
                message[3]))
    db_data.close()
    return db_data_result