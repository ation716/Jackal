import akshare as ak
import time
id_list=['002899','603215','002861']
# id='603037'
for i in range(10000):
    time.sleep(5)
# stock_intraday_em_df = ak.stock_intraday_em(symbol="603215")
    for id_ in id_list:
        df = ak.stock_bid_ask_em(symbol=id_)
        print(id_)
        print(df.iloc[8, 1],df.iloc[9, 1])
        print(df.iloc[10, 1],df.iloc[11, 1])
        print(df.iloc[20, 1],df.iloc[22, 1])
        print('-'*15)


# def