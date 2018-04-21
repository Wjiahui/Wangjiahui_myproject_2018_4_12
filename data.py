# -*- coding=utf8 -*-
"""
Fetch the daily stock prices from Google Finance for stocks in S & P 500.
@author: Wang Jiahui
"""
import click
import os
import pandas as pd
import random
import time
import urllib2

#from BeautifulSoup import BeautifulSoup
from datetime import datetime
#dir: directory 文件目录
DATA_DIR = "data"
RANDOM_SLEEP_TIMES = (1, 5)

# This repo "github.com/datasets/s-and-p-500-companies" has some other information about
# S & P 500 companies.
SP500_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents-financials.csv"
# 将多个路径组合后返回
SP500_LIST_PATH = os.path.join(DATA_DIR, "constituents-financials.csv")


def fetch_prices(symbol, out_name):
    """
    Fetch daily stock prices for stock `symbol`, since 1980-01-01.

    Args:
        symbol (str): a stock abbr. symbol, like "GOOG" or "AAPL".

    Returns: a bool, whether the fetch is succeeded.
    """
    # Format today's date to match Google's finance history api.
    now_datetime = datetime.now().strftime("%b+%d,+%Y")

    BASE_URL = "https://finance.google.com/finance/historical?output=csv&q={0}&startdate=Jan+1%2C+1980&enddate={1}"
    symbol_url = BASE_URL.format(
        urllib2.quote(symbol),
        urllib2.quote(now_datetime, '+')
    )
    print "Fetching {} ...".format(symbol)
    print symbol_url

    try:
        f = urllib2.urlopen(symbol_url)
        with open(out_name, 'w') as fin:
            print >> fin, f.read()
    except urllib2.HTTPError:
        print "Failed when fetching {}".format(symbol)
        return False

    data = pd.read_csv(out_name)
    if data.empty:
        print "Remove {} because the data set is empty.".format(out_name)
        os.remove(out_name)
    else:
        dates = data.iloc[:,0].tolist()
        print "# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0])

    # Take a rest
    sleep_time = random.randint(*RANDOM_SLEEP_TIMES)
    print "Sleeping ... %ds" % sleep_time
    time.sleep(sleep_time)
    return True

# click: 用于快速创建命令行
# command: 装饰一个函数，使之成为命令行接口
@click.command(help="Fetch stock prices data")
# option: 为其添加命令行选项
@click.option('--continued', is_flag=True)
def main(continued):
    # seed(): 随机数生成时所用算法开始的整数值
    random.seed(time.time())
    num_failure = 0

    symbols = ['GOOG']
    # 对于一个可迭代的对象，enumerate将其组成一个索引序列
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(DATA_DIR, sym + ".csv")
        if continued and os.path.exists(out_name):
            print "Fetched", sym
            continue

        succeeded = fetch_prices(sym, out_name)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print "# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure)


if __name__ == "__main__":
    main()
