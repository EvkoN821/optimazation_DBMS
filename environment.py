import time
import mysql.connector
import math
import os
import psutil
from datetime import datetime
from pyuac import main_requires_admin
from knobs_def import knobs_definition, knobs_for_config, knobs_default
from elevate import elevate


knobs_definition = knobs_definition
knobs_for_config = knobs_for_config
knobs_default = knobs_default
elevate()


class MysqlConnector:
    def __init__(self, host='localhost', user='root', passwd='root', name='employees'):
        super().__init__()
        self.dbhost = host
        self.dbuser = user
        self.dbpasswd = passwd
        self.dbname = name
        self.conn = None
        self.cursor = None
        self.connect_db()

    def connect_db(self):
        try:
            self.conn = mysql.connector.connect(host=self.dbhost, user=self.dbuser, passwd=self.dbpasswd, db=self.dbname)
            if self.conn:
                self.cursor = self.conn.cursor()
        except mysql.connector.Error as err:
            print(f"Error:  {err}")

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def fetch_results(self, sql, json=True):
        results = False
        try:
            if self.conn:
                self.cursor.execute(sql)
                results = self.cursor.fetchall()
                if json:
                    columns = [col[0] for col in self.cursor.description]
                    return [dict(zip(columns, row)) for row in results]
            return results
        except mysql.connector.Error as err:
            print(f"Error:  {err}")
            return False

    def execute(self, sql):
        if self.conn:
            self.cursor.execute(sql)

class MySQLEnv:
    def __init__(self, knobs_list, default_knobs):
        super().__init__()
        self.knobs_list = knobs_list
        self.states_list = []
        self.db_con = MysqlConnector()
        self.states = []
        self.score = 0.
        self.steps = 0
        self.terminate = False
        self.last_latency = 0
        self.default_latency = 0
        self.default_knobs = default_knobs

    def db_is_alive(self):
        try:
            flag = True
            while flag:
                for proc in psutil.process_iter():
                    if proc.name() == "mysqld.exe":
                        flag = False
                        break
                if flag:
                    time.sleep(20)
        except psutil.Error as err:
            print(f"Error:  {err}")


    # @main_requires_admin
    def apply_knobs(self, knobs):
        self.db_is_alive()
        db_conn = MysqlConnector()
        try:
            with open("C:\\ProgramData\\MySQL\\MySQL Server 8.0\\my.ini", "r") as f:
                x = [i for i in f]
            for i in range(len(knobs_definition)):
                if knobs_definition[i] in knobs_for_config:
                    x = self.apply_knobs_in_config(int(knobs[i]), knobs_definition[i], x)
                else:
                    try:
                        sql = f"SET GLOBAL {knobs_definition[i]} = {int(knobs[i])}"
                    except ValueError:
                        sql = f"SET GLOBAL {knobs_definition[i]} = {int(knobs_default[i])}"
                    try:
                        db_conn.execute(sql)
                    except:
                        sql = f"SET {knobs_definition[i]} = {int(knobs[i])}"
                        db_conn.execute(sql)
            db_conn.close_db()
            with open("C:\\ProgramData\\MySQL\\MySQL Server 8.0\\my.ini", 'w') as f:
                f.write(''.join(x))
            self.db_restart()
        except FileNotFoundError or PermissionError as err:
            print(f"Error:  {err}")

    def apply_knobs_in_config(self, knobs_val, knobs_name, x):
        if knobs_name in ''.join(x):
            for i in range(len(x)):
                if knobs_name in x[i] and x[i][0] != "#":
                    # print(x[i])
                    x[i] = x[i][:x[i].index('=') + 1] + f'{knobs_val}' + "\n"
        else:
            x.append(f'{knobs_name}={knobs_val}\n')
        return x

    # @main_requires_admin
    def db_restart(self):
        try:
            os.system("C:\\Windows\\System32\\net.exe stop MySql80")
            os.system("C:\\Windows\\System32\\net.exe start MySql80")
            #subprocess.call(["\"C:\\Program Files\\MySQL\\MySQL Server 8.0\\bin\\mysqladmin.exe\" -u root shutdown -p\"root\""])
            #subprocess.call(["\"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld\" --defaults-file=\"C:\Program Files\MySQL\MySQL Server 8.0\my.ini\" "])
        except OSError as err:
            print(f"Error:  {err}")

    def get_latency(self):
        count_q = 1
        self.db_con.connect_db()
        t1 = float(datetime.utcnow().strftime('%S.%f'))
        for i in range(count_q):
            # r = self.db_con.fetch_results("SELECT COUNT(*) FROM actor")
            r = self.db_con.fetch_results("SELECT * FROM employees e WHERE EXISTS (SELECT * FROM titles t WHERE t.emp_no = e.emp_no AND title = 'Assistant Engineer');")
        t2 = float(datetime.utcnow().strftime('%S.%f'))
        self.db_con.close_db()
        latency = math.fabs((t2-t1)) / count_q
        return latency

    def get_internal_metrics(self):
        internal_metrics = []
        db_conn = MysqlConnector()
        sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
        res = db_conn.fetch_results(sql)
        for i, v in enumerate(res):
            internal_metrics.append(v.get("COUNT"))
        return internal_metrics

    def get_states(self):
        external_metrics = self.get_latency()
        internal_metrics = self.get_internal_metrics()
        return external_metrics, internal_metrics

    def step(self, knobs):
        self.steps += 1
        self.apply_knobs(knobs)
        s = self.get_states()
        latency, internal_metrics = s
        reward = self.get_reward(latency)
        next_state = internal_metrics

        return next_state, reward, False, latency

    def init(self):
        self.score = 0.
        self.steps = 0
        self.terminate = False

        self.db_is_alive()
        self.apply_knobs(self.default_knobs)
        self.db_is_alive()
        s = self.get_states()
        latency, internal_states = s

        self.last_latency = latency
        self.default_latency = latency
        state = internal_states
        return state

    def get_reward(self, latency):

        def calculate_reward(delta0, deltat):
            a = 5
            if delta0 > 0 and deltat < 0:
                _reward = 0
            elif delta0>0 and deltat > 0:
                _reward = deltat * a**(delta0+1)
            else:
                _reward = (deltat-1) * a**(abs(delta0)+1)


            return _reward

        if latency == 0:
            return 0
        try:
            delta_0_lat = float((-latency + self.default_latency)) / self.default_latency
        except ZeroDivisionError:
            delta_0_lat = float((-latency + self.default_latency)) / 0.1**8
        try:
            delta_t_lat = float((-latency + self.last_latency)) / self.last_latency
        except ZeroDivisionError:
            delta_t_lat = float((-latency + self.last_latency)) / 0.1**8
        reward = calculate_reward(delta_0_lat, delta_t_lat)
        if reward < -10**3: reward = -10**3
        self.score += reward

        return reward