# 分时盯盘，
# 1. 需要检测是否有大单，是否有：
# 托单	买盘（买二~买四）	下方挂出巨额买单	制造支撑假象，掩护出货	高位警惕，低位观察
# 压单	卖盘（卖二~卖四）	上方挂出巨额卖单	制造压力假象，压制吸筹	低位勿怕，突破跟进
# 夹单	买盘和卖盘同时	上下均挂出大单，锁定股价	控制股价，上下洗盘	观望等待，跟随突破
# 空中成交	不显示在盘口	盘后显示巨量直接成交	对倒、换仓、传递信号	结合股价位置判断
# 扫货（扫单）	卖盘	连续大买单吃掉上方卖盘	强势拉升，突破压力	积极信号，考虑跟进
# 砸盘（砸单）	买盘	连续大卖单吃掉下方买盘	强势出货，制造恐慌	看空信号，果断离场
# 2. 检测最近变化幅度，分时箱线


import akshare as ak
import time
from collections import deque
# id_list=['002755','000633','603716','002728','600721']
# ask,hjtz,
# id_list=['002164','003015','002728','605228']
#nbdl,rjgd,tyyy,stkj
# id_list=['002164','003015','603738','600230','002178','600720']
#nbdl,rjgd,tjkj,czdh,yhzn
id_list=['603686','002298','600580','603767','600203','002600']
# flm,zdxl,wldq,zmcd,frdz,lyzz
# id_list=['002164','003015','603738','600230','002178']
dq_list=[deque(maxlen=12) for _ in range(len(id_list))]
vloume=[0 for _ in range(len(id_list))]



for i in range(10000):
    time.sleep(2)
# stock_intraday_em_df = ak.stock_intraday_em(symbol="603215")
    for i in range(len(id_list)):
        time.sleep(3/len(id_list))
        df = ak.stock_bid_ask_em(symbol=id_list[i])
        if not vloume[i]:
            vloume[i]=df.iloc[24, 1]
        dq_list[i].append((df.iloc[22, 1],df.iloc[24, 1]-vloume[i]))
        # print('x:', id_list[i][:5], df.iloc[20, 1], df.iloc[22, 1],df.iloc[24, 1])
        vloume[i]=df.iloc[24, 1]
        if len(dq_list[i])<2:
            volume_5s = -1
        else:
            volume_5s =  dq_list[i][-1][1]
        if len(dq_list[i])<5:
            volume_30s_qr=-1
        else:
            sum_30 =0
            for j in range(1,6):
                sum_30 += dq_list[i][-j][1]
            if sum_30==0:
                volume_30s_qr=0
            else:
                volume_30s_qr = dq_list[i][-1][1]/(sum_30/5)

        if len(dq_list[i])<12:
            volume_60s_qr=-1
        else:
            sum_60 =0
            for j in range(1,13):
                sum_60 += dq_list[i][-j][1]
            if sum_60==0:
                volume_60s_qr=0
            else:
                volume_60s_qr = dq_list[i][-1][1]/(sum_60/12)
        print('x:', id_list[i][:5], df.iloc[20, 1], df.iloc[22, 1],end=' ')
        if volume_30s_qr>1:
            print(round(volume_30s_qr,2),end=' ')
        if volume_60s_qr>1:
            print(round(volume_60s_qr,2),end=' ')
        print()
    print('-'*15)

# 需要看出承压能力