from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
import sqlite3 as sql
from rocket_learn.rollout_generator.redis.utils import MODEL_LATEST, VERSION_LATEST
from time import  sleep
import os


redis_info = {
        "host": os.environ["REDIS_HOST"] if "REDIS_HOST" in os.environ else "localhost",
        "password": os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else None,
    }


def create_redis(info: dict):
    return Redis(host=info["host"],
                 password=info["password"],
                 retry_on_error=[ConnectionError, TimeoutError],
                 retry=Retry(ExponentialBackoff(cap=10, base=1), 25))


def sql_connection(cache_name: str):
    sql_conn = sql.connect('redis-model-cache-' + cache_name + '.db')
    # if the table doesn't exist in the database, make it
    sql_conn.execute("""
                    CREATE TABLE if not exists MODELS (
                        id TEXT PRIMARY KEY,
                        parameters BLOB NOT NULL
                    );
                """)
    return sql_conn


def model_caretaker(sql_conn, red, version):
    models = sql_conn.execute("SELECT parameters FROM MODELS WHERE id == ?", (version,)).fetchall()
    if len(models) == 0:
        bytestream = red.get(MODEL_LATEST)
        sql_conn.execute('INSERT INTO MODELS (id, parameters) VALUES (?, ?)', (version, bytestream))
        sql_conn.commit()

        previous_check = sql_conn.execute("SELECT parameters FROM MODELS WHERE id == ?", (version + 2,)).fetchall()
        if len(previous_check) > 0:
            sql_conn.execute(f"DELETE FROM MODELS WHERE id == '{version + 2}'")
            sql_conn.commit()


def main():
    print("model updater started")
    r = create_redis(redis_info)
    sq = sql_connection("raptor_model_database")
    latest_local = int(r.get(VERSION_LATEST))
    model_caretaker(sq, r, latest_local)
    print("model updated!")
    while True:
        available_version = int(r.get(VERSION_LATEST))
        if latest_local != available_version:
            model_caretaker(sq, r, available_version)
            latest_local = available_version
            print("model updated!")
        sleep(10)


if __name__ == "__main__":
    main()

