import ccxt
ym=None
yE=True
yz=object
yR=dir
yU=str
ys=input
yV=enumerate
ya=zip
yP=abs
yw=int
yd=len
yn=range
yY=max
yH=False
yh=float
yq=getattr
yp=map
yc=open
yl=round
Wc=ccxt.bittrex
Wp=ccxt.poloniex
import time
Wl=time.time
WX=time.sleep
import tensorflow as tf
Wf=tf.Session
Wb=tf.train
Wv=tf.float32
WC=tf.placeholder
WM=tf.Graph
import numpy as np
Wj=np.yl
Wu=np.subtract
Wx=np.set_printoptions
WT=np.stack
Wt=np.zeros
WN=np.squeeze
import socketio
import pause
WS=pause.minutes
WQ=pause.seconds
from flask import Flask,request,jsonify,session
We=request.sid
WL=request.json
from datetime import datetime,timedelta
Wk=datetime.now
yK=datetime.utcnow
WD=datetime.time
Wo=datetime.date
from flask_cors import CORS
import backtrader as bt
yW=bt.SignalStrategy
yI=bt.TimeFrame
yi=bt.num2date
yG=bt.sizers
yA=bt.Order
yr=bt.feeds
yF=bt.brokers
yJ=bt.ind
yg=bt.Cerebro
yB=bt.SIGNAL_LONGSHORT
from flask_socketio import SocketIO,emit
K=Flask(__name__)
F=SocketIO(K)
W=ym
y=yE
J=CORS(K)
class WH(yz):
 def __init__(B,A,i):
  B.exchanges=[A,i]
def WF(yR,U,a,f,M,KP):
 with Wf(graph=WM())as G:
  print(We)
  F.emit('logs','Loading model...',namespace='/test',room=KP)
  r=Wb.import_meta_graph('altmodel/1/netfile.meta')
  r.restore(G,save_path='altmodel/1/netfile')
  m=WC(Wv,shape=[3,11,1])
  E=G.graph.get_operation_by_name('Adam')
  z=G.graph.get_tensor_by_name('Softmax:0')
  ys=G.graph.get_tensor_by_name('Placeholder:0')
  i=G.graph.get_tensor_by_name("Placeholder_1:0")
  i1=G.graph.get_tensor_by_name("Placeholder_2:0")
  i2=G.graph.get_tensor_by_name("Placeholder_3:0")
  R=[E,z]
  U=U.reshape(1,3,11,31)
  U=U/U[:,:,-1,0,ym,ym]
  Wx(suppress=yE)
  d2=Wt((1,11))
  la="last w:"+' '+yU(a)
  F.emit('logs',la,namespace='/test',room=KP)
  w=WN(G.Wn(z,feed_dict={ys:1,i:U,i1:a}))
  w=w[-11:]
  V=Wu(a,w).tolist()[0]
  print(V)
  t='Transaction Vector: '+yU(V)
  F.emit('logs',t,namespace='/test',room=KP)
  a=V
  Wy(f,V,M,a,KP)
def Wy(f,V,M,a,KP):
 print(M,'exchange')
 print(V)
 P=M.fetch_balance()
 P=[P[x]['free']for x in f]
 print(P)
 w=[]
 Ws=f
 for n in Ws:
  if n=='USDT':
   n='BTC/USDT'
   Y=M.fetch_ticker(n)
   print(Y)
   w.append(Y['info']['lowestAsk'])
  else:
   n=n+'/BTC'
   Y=M.fetch_ticker(n)
   print(Y)
   w.append(Y['info']['lowestAsk'])
 print(w)
 H=[]
 print(V)
 for h,(n,c)in yV(ya(f,V)):
  print(c)
  if c>0:
   if n=="USDT":
    q="BTC/USDT"
    p=M.create_order(q,amount=c,Ww=w[h],side='buy',type='limit')
    F.emit('logs',yU(p),namespace='/test',room=KP)
   else:
    q=n+'/BTC'
    p=M.create_order(q,amount=c,Ww=w[h],side='buy',type='limit')
    F.emit('logs',yU(p),namespace='/test',room=KP)
  if c<0:
   c=yP(c)
   if n=="USDT":
    q="BTC/USDT"
    l=M.create_order(q,amount=c,Ww=w[h],side='sell',type='limit')
    F.emit('logs',yU(l),namespace='/test',room=KP)
   else:
    q=n+'/BTC'
    l=M.create_order(q,amount=c,Ww=w[h],side='sell',type='limit')
    F.emit('logs',yU(l),namespace='/test',room=KP)
 WS(30)
 U,X,f,M=WJ(M,a)
 X=X.reshape(1,-1)
 b=WF('1',U,X,f,M)
def WJ(M,KP,last_w=ym):
 C=Wk()-timedelta(hours=15,minutes=30)
 print(C)
 C=C.strftime("%Y-%m-%d %H:%M:%S")
 v=M.parse8601(C)
 x=M.fetch_ohlcv('ETH/BTC',timeframe='30m',since=v,limit=37)
 t=M.fetch_ohlcv('LTC/BTC',timeframe='30m',since=v,limit=37)
 N=M.fetch_ohlcv('XRP/BTC',timeframe='30m',since=v,limit=37)
 print(N)
 u=M.fetch_ohlcv('BTC/USDT',timeframe='30m',since=v,limit=37)
 print(u)
 for li in u:
  li[:]=[1/x for x in li]
 T=M.fetch_ohlcv('ETC/BTC',timeframe='30m',since=v,limit=37)
 j=M.fetch_ohlcv('DASH/BTC',timeframe='30m',since=v,limit=37)
 S=M.fetch_ohlcv('XMR/BTC',timeframe='30m',since=v,limit=37)
 Q=M.fetch_ohlcv('XEM/BTC',timeframe='30m',since=v,limit=37)
 e=M.fetch_ohlcv('FCT/BTC',timeframe='30m',since=v,limit=37)
 L=M.fetch_ohlcv('GNT/BTC',timeframe='30m',since=v,limit=37)
 k=M.fetch_ohlcv('ZEC/BTC',timeframe='30m',since=v,limit=37)
 f=[x,t,N,u,T,j,S,Q,e,L,k]
 o=['ETH','LTC','XRP','USDT','ETC','DASH','XMR','XEM','FCT','GNT','ZEC']
 F.emit('logs',yU(o),namespace='/test',room=KP)
 D=-1
 L=[]
 for n in f:
  D+=1
  for KF in n:
   KW=o[D]
   Ky=KF[0]
   KJ=KF[2]
   KB=KF[3]
   KA=KF[4]
   L.append([KW,Ky,KJ,KB,KA])
 import pandas as pd
 yO=pd.DataFrame
 df=yO(L,columns=['coin','date','low','high','close'])
 df=df.drop(['date'],axis=1)
 if last_w==ym:
  last_w=[[-0.98176,0.018265,0.01821,0.018255,-0.981775,0.01824,0.01822,0.018235,0.01827003,0.018145,0.01816]]
 Ki=df[df.coin=='ETH']
 KI=df[df.coin=='LTC']
 Kg=df[df.coin=='XRP']
 Kr=df[df.coin=='USDT']
 KG=df[df.coin=='ETC']
 KO=df[df.coin=='DASH']
 Km=df[df.coin=='XMR']
 KE=df[df.coin=='XEM']
 Kz=df[df.coin=='FCT']
 KR=df[df.coin=='GNT']
 KU=df[df.coin=='ZEC']
 Ki=Ki.drop(['coin'],axis=1).iloc[-31:]
 KI=KI.drop(['coin'],axis=1).iloc[-31:]
 Kg=Kg.drop(['coin'],axis=1).iloc[-31:]
 Kr=Kr.drop(['coin'],axis=1).iloc[-31:]
 KG=KG.drop(['coin'],axis=1).iloc[-31:]
 KO=KO.drop(['coin'],axis=1).iloc[-31:]
 Km=Km.drop(['coin'],axis=1).iloc[-31:]
 KE=KE.drop(['coin'],axis=1).iloc[-31:]
 Kz=Kz.drop(['coin'],axis=1).iloc[-31:]
 KR=KR.drop(['coin'],axis=1).iloc[-31:]
 KU=KU.drop(['coin'],axis=1).iloc[-31:]
 li=[Ki,KI,Kg,Kr,KG,KO,Km,KE,Kz,KR,KU]
 for l in li:
  print(l.shape)
 U=WT((Ki.values,KI.values,Kg.values,Kr.values,KG.values,KO.values,Km.values,KE.values,Kz.values,KR.values,KU.values))
 U=U.reshape(3,11,31)
 return U,last_w,o,M
@K.route('/')
def WB():
 return "grettings wanderer"
@F.on('rl',namespace='/test')
def WA(message):
 print(message,'this is the message')
 print(We)
 F.emit('logs','STARTING BOT...',namespace='/test',room=We)
 Ks=message['KEY']
 KV=message['SECRET']
 Ka={'apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*1000000000))}
 M=Wp(Ka)
 KP=We
 P=M.fetch_balance()
 Kw('balances',yU(P),namespace='/test',room=We)
 print(M.secret,M.apiKey)
 Kd=Wk()
 Kn=Kd-timedelta(hours=8,minutes=30)
 U,X,f,M=WJ(M,KP)
 F.emit('coins',yU(f),namespace='/test',room=We)
 print(X)
 b=WF('1',U,X,f,M,KP)
'''
BEGIN ARBITRAGE
'''
class Wh(yz):
 def __init__(B):
  B.exchange=ym
  B.config=ym
  B.value=ym
def Wi(a,b):
 KY=((a-b)/a)*100
 return KY
def WI(alist,wanted_parts=1):
 KH=yd(alist)
 return[alist[i*KH//wanted_parts:(i+1)*KH//wanted_parts]for i in yn(wanted_parts)]
def Wg(w,bases,yY,o,Fr,KD):
 global W,y,KT
 O=Wh()
 O1=Wh()
 WJ=[]
 Kh=yd(w)
 Kq=[]
 for t in w[:Kh]:
  print(t)
  t=[c for item in t for c in item]
  print(t)
  TT=t[1::4]
  Kq.append(t[2::4][0])
  Kp=t[3::4]
  for Kc,Fz in ya(TT,Kp):
   try:
    try:
     Y=Kc.fetch_ticker(Fz)
     WJ.append(Y['last'])
     print((Y,'this is the ticker'))
    except:
     Y=Kc.fetch_ticker(Fz)
     WJ.append(Y['info']['Last'])
     print((Y,'this is the ticker'))
   except:
    print('something_wrong')
    F.emit('logs',yE,namespace='/test',broadcast=yE)
    F.emit('logs','One or more pairs unavailable',namespace='/test',broadcast=yE)
    y=yH
  print(WJ)
  Kl=""
  for p in WJ:
   Ws=yU(p)
   Kl+=Ws+'\n'
  em='Prices($): '+Kl
  F.emit('logs',em,namespace='/test',broadcast=yE)
  WX(2)
 if y==yE:
  KX,Kf=WI(WJ,Kh)
  for KM,(v,v1)in yV(ya(KX,Kf)):
   print(v,v1)
   D=Wi(v,v1)
   print(D)
   D=Wj(D,decimals=3)
   F.emit('logs','Difference: '+yU(D)+'%'+' [Value 1: $'+yU(v)+' Value 2: $'+yU(v1)+']',namespace='/test',broadcast=yE)
   WQ(3)
   if yP(D)>yh(Fr):
    if v>v1:
     Kb=bases[KM]
     print(Kb)
     KC=Kq[1]
     Kv=Kq[0]
     Kx=KC.describe()['fees']
     Kt=Kx['trading']['maker']
     KN=Kx['trading']['taker']
     wi=Kx['funding']['withdraw']
     print(wi)
     try:
      Ku=Kx['funding']['withdraw'][Kb]
      KT=v1+(v1*Kt)+Ku
      Kj=Kb+'/USDT'
      try:
       W=Kv.fetch_deposit_address(Kb)
       print(W)
       U='Wallet Address for Transfer:'+yU(W)
       F.emit('logs',U,namespace='/test',broadcast=yE)
      except:
       try:
        W=Kv.create_deposit_address(Kb)
        print(W)
        W=W['address']
        print(W)
        U='Wallet Address for Transfer:'+yU(W)
        F.emit('logs',U,namespace='/test',broadcast=yE)
       except:
        F.emit('logs',yE,namespace='/test',broadcast=yE)
        F.emit('logs','Exchange does not allow wallet creation via API, or API down',namespace='/test',broadcast=yE)
        continue
      KS=(Wi(v,KT))
      print((Kt,KN,Ku))
      if KT and v>v1:
       try:
        KC.create_limit_buy_order(Kj,yY,v*.001)
        WP=KC.fetch_balance()[Kb]
        p="Starting buy order on"+yU(o[1])+'for'+yU(Kb)
        F.emit('logs',p,namespace='/test',broadcast=yE)
        KC.withdraw(Kb,WP,W)
        WX(3)
        KQ=Kv.fetch_balance()[Kb]
        Wr(Kv,Kb,Kj,WP,KQ,o[0])
       except:
        F.emit('logs',"Problem parsing deposit address, or  Not enough funds",namespace='/test',)
     except:
      Ku=Kx['funding']['withdraw']
      D="Can't dynamically parse withdrawl fees, here are the currencies we can:"+yU(Ku)
      F.emit('logs',D,namespace='/test',broadcast=yE)
    elif v1>v:
     Kb=bases[KM]
     print(Kb)
     KC=Kq[0]
     Kv=Kq[1]
     Kx=KC.describe()['fees']
     Kt=Kx['trading']['maker']
     KN=Kx['trading']['taker']
     wi=Kx['funding']['withdraw']
     print(wi)
     try:
      Ku=Kx['funding']['withdraw'][Kb]
      KT=v+(v*Kt)+Ku
      Kj=Kb+'/USDT'
      try:
       W=Kv.fetch_deposit_address(Kb)
       W=W['address']
       print(W)
       U='Wallet Address for Transfer:'+yU(W)
       F.emit('logs',U,namespace='/test',broadcast=yE)
      except:
       try:
        W=Kv.create_deposit_address(Kb)
        W=W['address']
        print(W)
        U='Wallet Address for Transfer:'+yU(W)
        F.emit('logs',U,namespace='/test',broadcast=yE)
       except:
        F.emit('logs',yE,namespace='/test',broadcast=yE)
        F.emit('logs','Exchange does not allow wallet creation via API, or API down',namespace='/test',broadcast=yE)
      KS=(Wi(v,KT))
      print((Kt,KN,Ku))
     except:
      Ku=Kx['funding']['withdraw']
      D="Can't dynamically parse withdrawl fees, here are the currencies we can:"+yU(Ku)
      F.emit('logs',yE,namespace='/test',broadcast=yE)
      F.emit('logs',D,namespace='/test',broadcast=yE)
      if KT and v1>KT*.01:
       try:
        KC.create_limit_buy_order(Kj,yY,v*.001)
        WP=KC.fetch_balance()[Kb]
        p="Starting buy order on"+yU(o[0])+'for'+yU(Kb)
        F.emit('logs',p,namespace='/test',broadcast=yE)
        KC.withdraw(Kb,WP,W)
        WX(3)
        KQ=Kv.fetch_balance()[Kb]
        Wr(Kv,Kb,Kj,WP,KQ,o[1])
       except:
        F.emit('logs',yE,namespace='/test',broadcast=yE)
        F.emit('logs',"Problem parsing deposit address, or  Not enough funds",namespace='/test',broadcast=yE)
 print(Kq)
 WX(5)
 Wg(w,bases,yY,o,Fr,KD)
 return 'something'
Ke=0
def Wr(trader,Kb,Kj,amount,KL,exch):
 global Ke
 Ke=trader.fetch_balance()[Kb]
 while Ke==KL:
  Ke=trader.fetch_balance()[Kb]
 E="funds arrived at"+yU(exch)
 F.emit('logs',E,namespace='/test',broadcast=yE)
 Kk=yU(Kb)+'/USDT'
 Ww=trader.fetch_ticker(Kk)['last']*.001
 trader.create_limit_sell_order(Kj,Ke,Ww)
 A='Selling'+' '+yU(Kb)+' '+"to USDT"
 F.emit('logs',A,namespace='/test',broadcast=yE)
@K.route('/balance_arbi',methods=['POST'])
def WG():
 U=WL
 KD,FK,FW,Kp,Fy=U["EXCHANGES"],U["KEYS"],U["SECRETS"],U["CURRENCY"],U["USD"]
 KD=[item for items in KD for item in items.split(",")]
 FK=[item for items in FK for item in items.split(",")]
 FW=[item for items in FW for item in items.split(",")]
 FJ=[]
 for h,(Ks,KV)in yV(ya(FK,FW)):
  if KD[h]=='bitfinex':
   FB={'apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*100000))}
   FJ.append(FB)
  else:
   FB={'apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*1000))}
   FJ.append(FB)
 FA=[]
 for i,c in yV(FJ,WE=0):
  if KD[i]=='bitfinex':
   Fi=yq(ccxt,'bitfinex2')
   FA.append(Fi(c))
  else:
   Fi=yq(ccxt,KD[i])
   FA.append(Fi(c))
 FI=[]
 for ex in FA:
  b=ex.fetch_balance()
  c=[]
  print(b)
  for h,(Ks,value)in yV(b.items()):
   try:
    print(value)
    WQ(3)
    if 'free' in value.keys():
     if value['free']>0.0:
      Fg=' '+yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
      c.append(Fg)
     else:
      c.append('Wallets are empty')
   except:
    print('here')
  FI.append(c)
 di={}
 di.setdefault('exchangeA',[])
 di.setdefault('exchangeB',[])
 for i,bal in yV(FI):
  if i==0:
   di['exchangeA']=bal
  else:
   di['exchangeB']=bal
 return jsonify(di)
@K.route('/arbitrage',methods=['POST'])
def WO():
 F.emit('logs','STARTING BOT...',namespace='/test',broadcast=yE)
 U=WL
 KD,FK,FW,Kp,Fy,Fr=U["EXCHANGES"],U["KEYS"],U["SECRETS"],U["CURRENCY"],U["USD"],U["TRADE_PERC"]
 KD=[item for items in KD for item in items.split(",")]
 FK=[item for items in FK for item in items.split(",")]
 FW=[item for items in FW for item in items.split(",")]
 Kp=[item for items in Kp for item in items.split(",")]
 FG=yd(KD)
 FJ=[]
 print(Fr)
 FO=WH(KD[0],KD[1])
 Fr=Fr[0]
 print(KD,'exchanges')
 for h,(Ks,KV)in yV(ya(FK,FW)):
  if KD[h]=='bitstamp':
   FB={'uid':'mzxy9253','apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*1000))}
  else:
   FB={'apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*1000000))}
  FJ.append(FB)
 FA=[]
 for i,c in yV(FJ,WE=0):
  Fi=yq(ccxt,KD[i])
  FA.append(Fi(c))
 Fm=[]
 print(FA,'indicators')
 FE=ym
 for n,c in ya(KD,FJ):
  print(c)
  if n=='kraken':
   FE=yF.CCXTBroker(exchange=n,currency='USD',config=c)
  else:
   FE=yF.CCXTBroker(exchange=n,currency='USDT',config=c)
  Fm.append(FE)
 w=[]
 for i,c in yV(FA):
  print(i)
  Y=ym
  for h,Fz in yV(Kp):
   a=Fz
   if 'kraken'==KD[1]:
    Fz=yU(Fz)+'/USD'
    Y=FA[i]
    w.append([a,Y,c,Fz])
   else:
    Fz=yU(Fz)+'/USDT'
    Y=FA[i]
    w.append([a,Y,c,Fz])
 FR=WI(w,FG)
 print(FR)
 t=Wg(FR,Kp,Fy,KD,Fr,FO)
'''
BEGIN Ema
'''
Ks=''
KV=''
FU=ym
@F.on('ema',namespace='/test')
class Wq(yW):
 global FE,Fk,helper
 global Fd
 Fs=0
 FV=(('stop_loss',0.1),('take_profit',0.2),('low',14),('high',90))
 def Wm(B,txt,dt=ym):
  global FE,Fk,Fd
  ''' Logging function for this strategy'''
  dt=dt or B.datas[0].Wo(0)
  Fa=B.datas[0].WD()
  print('%s - %s, %s'%(dt.isoformat(),Fa,txt))
 def __init__(B):
  global FE,Fk,Fd,M
  global FB
  B.dataclose=B.datas[0].close
  FP,Fw=yJ.EMA(period=B.p.low),yJ.EMA(period=B.p.high)
  B.signal_add(yB,yJ.CrossUp(FP,Fw))
  B.signal_add(yB,yJ.CrossDown(FP,Fw))
  B.crossover=yJ.CrossOver(FP,Fw)
  B.crossup=yJ.CrossUp(FP,Fw)
  B.crossdown=yJ.CrossDown(FP,Fw)
  B.Wd=yJ.MomentumOscillator(period=B.p.high)
  if FB!={}:
   print(yU(FB)+'this is it')
   if M=='bitfinex':
    Fd=yq(ccxt,'bitfinex2')
    Fd.trades=0
    Fd.orders=ym
   elif M=='hitbtc':
    Fd=yq(ccxt,'hitbtc2')
    Fd.trades=0
    Fd.orders=ym
   else:
    Fd=yq(ccxt,M)
    Fd.trades=0
    Fd.orders=ym
  B.buyprice=ym
  B.buycomm=ym
  B.order=ym
  B.signal=0
  B.price_at_signal=0
  B.trades=0
 def WE(B):
  if FB!={}:
   Fd.trades=0
   Fn=0
 def Wz(B,trade):
  global FE,Fk,Fd
  if not trade.isclosed or Fd.trades:
   return
  Wm= B.Wm('OPERATION PROFIT, GROSS %.2f, NET %.2f'%(trade.pnl,trade.pnlcomm))
  F.emit('logs',Wm,namespace='/test',broadcast=yE)
  return Wm
 def WR(B,order):
  global FE,Fk,Fd
  if order.status in[order.Margin,order.Rejected]:
   pass
  if order.status in[order.Submitted,order.Accepted]:
   return
  elif order.status==order.Cancelled:
   Wm=B.Wm(' '.join(yp(yU,['CANCEL ORDER. Type :',order.info['name'],"/ DATE :",B.data.num2date(order.executed.dt).date().isoformat(),"/ PRICE :",order.executed.price,"/ SIZE :",order.executed.size,])))
   F.emit('logs',Wm,namespace='/test',broadcast=yE)
   return Wm
  elif order.status==order.Completed:
   if 'name' in order.info:
    Wm=B.Wm("%s: REF : %s / %s / PRICE : %.3f / SIZE : %.2f / COMM : %.2f"%(order.info['name'],order.ref,B.data.num2date(order.executed.dt).date().isoformat(),order.executed.price,order.executed.size,order.executed.comm))
    F.emit('logs',Wm,namespace='/test',broadcast=yE)
    return Wm
   else:
    if order.isbuy():
     FY=order.executed.price*(1.0-B.params.stop_loss)
     FH=order.executed.price*(1.0+B.params.take_profit)
     Fh=(FE.getcash()*0.5)
     Fq=Fh/B.data.close[0]
     Fp=Fd.sell(exectype=yA.StopTrailLimit,Ww=FY,size=Fq)
     Fp.addinfo(name="STOP")
     Fc=Fd.sell(exectype=yA.StopTrailLimit,Ww=FH,size=Fq,oco=Fp)
     Fc.addinfo(name="PROFIT")
     Wm=B.Wm("SignalPrice : %.3f Buy: %.3f, Stop: %.3f, Profit : %.3f"%(B.price_at_signal,order.executed.price,FY,FH))
     F.emit('logs',Wm,namespace='/test',broadcast=yE)
     return Wm
    elif order.issell():
     FY=order.executed.price*(1.0+B.params.stop_loss)
     FH=order.executed.price*(1.0-B.params.take_profit)
     Fl=(Fk.getcash()*0.5)
     FX=Fl/B.data.close[0]*-1
     Fp=Fd.buy(exectype=yA.StopTrailLimit,Ww=FY,size=FX)
     Fp.addinfo(name="STOP")
     Fc=Fd.buy(exectype=yA.StopTrailLimit,Ww=FH,size=FX,oco=Fp)
     Fc.addinfo(name="PROFIT")
     Wm=B.Wm("SignalPrice: %.3f Sell: %.3f, Stop: %.3f, Profit : %.3f"%(B.price_at_signal,order.executed.price,FY,FH))
     F.emit('logs',Wm,namespace='/test',broadcast=yE)
     return Wm
 def WU(B):
  global FE,Fk,Fd,FU
  for U in B.datas:
   print('*'*5,'NEXT:',yi(U.datetime[0]),U._name,U.yc[0],U.high[0],U.low[0],U.close[0],U.volume[0],yI.getname(U._timeframe),yd(U))
   Ff=('*'*5,'NEXT:',yi(U.datetime[0]),U._name,U.yc[0],U.high[0],U.low[0],U.close[0],U.volume[0],yI.getname(U._timeframe),yd(U))
   FM=""
   Ky='Date: '+yU(yi(U.datetime[0]).strftime('%Y-%m-%d'))
   yc='Open: '+yU(U.yc[0])
   KJ='Low: '+yU(U.low[0])
   KB='High: '+yU(U.high[0])
   Fb='Volume: '+yU(U.volume[0])
   FC=Ky+'\n'+yc+'\n'+KJ+'\n'+KB+'\n'+Fb
   FM+=FC
   F.emit('logs',FM,namespace='/test',broadcast=yE)
   print('binanceUSDT  Value: ',FE.getcash())
   Fv='Exchange USDT: '+yU(FE.getcash())
   Fx='BTC: '+yU(Fk.getcash())
   FU='EMA Trend Value: '+yU(B.Wd[0])
   Kl=""
   Kl+=Fv+'\n'
   Kl+=Fx+'\n'
   F.emit('logs',Kl,namespace='/test',broadcast=yE)
   print(B.Wd[0])
   Fh=(FE.getcash()*0.5)
   Fq=Fh
   Fl=(Fk.getcash()*0.5)
   FX=Fl
   Ft=(Fq*B.data.close[0])*(1-0.2)
   FH=(Fq*B.data.close[0])*(1+0.3)
   FN=(Fq*B.data.close[0])*(1+0.2)
   if not Fd.trades:
    if B.crossup:
     B.Wm('CrossUp')
     Fd.create_order(symbol='BTC/USDT',type='LIMIT',side='BUY',amount=15,Ww=yl(B.data.close[0],1),params={'timeInForce':'GTC','quantity':1,'price':B.data.close[0]})
     if B.Wd[0]<B.Wd[-1]and B.Wd[-2]:
      B.Wm('Greedy CrossUp')
      Fd.create_order(symbol='BTC/USDT',type='LIMIT',side='BUY',amount=15,Ww=yl(B.data.close[0],1),params={'timeInForce':'GTC','quantity':1,'price':B.data.close[0]})
    elif B.crossdown:
     B.Wm('Crossdown')
     Fd.create_order(symbol='BTC/USDT',type='LIMIT',side='SELL',amount=.0018,Ww=yl(B.data.close[0],1),params={'timeInForce':'GTC','quantity':1,'price':B.data.close[0]})
     if B.Wd[0]>B.Wd[-1]and B.Wd[-2]:
      B.Wm('Greedy Crossdown')
      Fd.create_order(symbol='BTC/USDT',type='LIMIT',side='SELL',amount=.0018,Ww=yl(B.data.close[0],1),params={'timeInForce':'GTC','quantity':1,'price':B.data.close[0]})
    else:
     return
@F.on('connect',namespace='/test')
def Ws():
 print('connect + thats sid')
 KP=We
 F.emit('connected',namespace='/test',broadcast=yE)
 KP='this is sid'+yU(KP)
 F.emit('connected',KP,namespace='/test',broadcast=yE)
 print(KP)
 return KP
@K.route('/fries',methods=['GET'])
def WV():
 Fu=Wc()
 Fx=yU(Fu.fetch_ticker('BTC/USDT')['last'])
 Ki=yU(Fu.fetch_ticker('ETH/USDT')['last'])
 Kg=yU(Fu.fetch_ticker('XRP/USDT')['last'])
 KI=yU(Fu.fetch_ticker('LTC/USDT')['last'])
 FT=yU(Fu.fetch_ticker('BCH/USDT')['last'])
 Fx=Fx[0:6]
 Ki=Ki[0:6]
 Kg=Kg[0:6]
 KI=KI[0:6]
 FT=FT[0:6]
 p=[Fx,Ki,Kg,KI,FT]
 return jsonify(p)
@K.route('/balances_rl',methods=['POST'])
def Wa():
 FB=ym
 print(yw(Wl()))
 if WL['EXCHANGE']=='bitfinex':
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*10001))}
 elif WL['EXCHANGE']=='hitbtc':
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*10001))}
 elif WL['EXCHANGE']=='poloniex':
  FB={'apiKey':WL['KEY'],'secret':WL['SECRET'],'nonce':lambda:yU(yw(Wl()*1000000000))}
 else:
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*1000))}
 Fi=ym
 if WL['EXCHANGE']=='bitfinex':
  Fi=yq(ccxt,'bitfinex2')
 if WL['EXCHANGE']=='hitbtc':
  Fi=yq(ccxt,'hitbtc2')
 else:
  Fi=yq(ccxt,WL['EXCHANGE'])
 Fj=Fi(FB)
 b=Fj.fetch_balance()
 c=[]
 print(b)
 if WL['EXCHANGE']=='poloniex':
  for Ks,value in b.items():
   if 'free' in value.keys():
    if value['free']>0.0:
     Fg=' '+yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
     c.append(Fg)
 elif WL['EXCHANGE']=='bitfinex':
  WQ(10)
  for Ks,value in b.items():
   print(value)
   if value!=[]and value!={}:
    Fg=yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
    c.append(Fg)
   else:
    c='Wallets are empty or API Issue'
 elif WL['EXCHANGE']=='hitbtc':
  WQ(10)
  for Ks,value in b.items():
   print(value)
   if value!=[]and value!={}:
    Fg=yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
    c.append(Fg)
   else:
    c='Wallets are empty or API Issue'
 else:
  for Ks,value in b.items():
   if 'Balance' in value.keys():
    if value['Balance']>0.0:
     Fg=yU(Ks)+': '+yU('{0:.5f}'.format(value['Balance']))
     c.append(Fg)
 print(c)
 return jsonify(c)
@K.route('/balances',methods=['POST'])
def WP():
 FB=ym
 print(yw(Wl()))
 if WL['EXCHANGE']=='bitfinex':
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*10001))}
 elif WL['EXCHANGE']=='hitbtc':
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*10001))}
 elif WL['EXCHANGE']=='poloniex':
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*1000000000))}
 else:
  FB={'apiKey':WL['API_KEY'],'secret':WL['API_SECRET'],'nonce':lambda:yU(yw(Wl()*1000))}
 Fi=ym
 if WL['EXCHANGE']=='bitfinex':
  Fi=yq(ccxt,'bitfinex2')
 if WL['EXCHANGE']=='hitbtc':
  Fi=yq(ccxt,'hitbtc2')
 else:
  Fi=yq(ccxt,WL['EXCHANGE'])
 Fj=Fi(FB)
 b=Fj.fetch_balance()
 c=[]
 print(b)
 if WL['EXCHANGE']=='poloniex':
  for Ks,value in b.items():
   if 'free' in value.keys():
    if value['free']>0.0:
     Fg=' '+yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
     c.append(Fg)
 elif WL['EXCHANGE']=='bitfinex':
  WQ(10)
  for Ks,value in b.items():
   print(value)
   if value!=[]and value!={}:
    Fg=yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
    c.append(Fg)
   else:
    c='Wallets are empty or API Issue'
 elif WL['EXCHANGE']=='hitbtc':
  WQ(10)
  for Ks,value in b.items():
   print(value)
   if value!=[]and value!={}:
    Fg=yU(Ks)+': '+yU('{0:.5f}'.format(value['free']))
    c.append(Fg)
   else:
    c='Wallets are empty or API Issue'
 else:
  for Ks,value in b.items():
   if 'Balance' in value.keys():
    if value['Balance']>0.0:
     Fg=yU(Ks)+': '+yU('{0:.5f}'.format(value['Balance']))
     c.append(Fg)
 print(c)
 return jsonify(c)
@F.on('/prices',namespace='/test')
def Ww():
 Fu=Wc()
 Fu.fetch_balance()
 Fx=yU(Fu.fetch_ticker('BTC/USDT')['last'])
 Ki=yU(Fu.fetch_ticker('ETH/USDT')['last'])
 Kg=yU(Fu.fetch_ticker('XRP/USDT')['last'])
 Fx=Fx[:6]
 Ki=Ki[:6]
 Kg=Kg[:6]
 p=[Fx,Ki,Kg]
 F.emit(p,namespace='/test',broadcast=yE)
 return p
@K.route('/trend',methods=['GET'])
def Wd():
 global FU
 return jsonify(FU)
FS=ym
@F.on('runner',namespace='/test')
def Wn(FQ):
 global cross,helper,M
 global FE,Fk,FB,FS
 while FQ['API_KEY']!='':
  Ks=FQ['API_KEY']
  KV=FQ['API_SECRET']
  M=FQ['EXCHANGE']
  F.emit('logs','Starting bot...',namespace='/test',broadcast=yE)
  print(Ks,KV)
  print('they are above')
  if Ks!='':
   print('got here')
   Fe=yg()
   FL=yK()-timedelta(minutes=240)
   if M=='poloniex' or 'bitfinex':
    FB={'apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*1000000000))}
   else:
    FB={'apiKey':Ks,'secret':KV,'nonce':lambda:yU(yw(Wl()*1000))}
   F.emit('ema',namespace='/test')
   FE=yF.CCXTBroker(exchange=M,currency='USDT',config=FB)
   FE=FE
   F.emit('logs',yU(FE.getcash())+' - USDT BALANCE',namespace='/test',broadcast=yE)
   Fk=yF.CCXTBroker(exchange=M,currency='BTC',config=FB)
   F.emit('logs',yU(Fk.getcash())+' - BTC BALANCE',namespace='/test',broadcast=yE)
   if M=='poloniex':
    FS=yr.CCXT(exchange=M,symbol="BTC/USDT",timeframe=yI.Minutes,compression=5,config=FB)
   elif M=='bitfinex':
    FS=yr.CCXT(exchange=M,symbol="BTC/USD",timeframe=yI.Minutes,compression=5,config=FB)
   elif M=='gateio':
    FS=yr.CCXT(exchange=M,symbol="BTC/USD",config=FB)
   else:
    FS=yr.CCXT(exchange=M,symbol="BTC/USDT",timeframe=yI.Minutes,compression=1,config=FB)
   Fo=FE.getcash()
   Fe.adddata(FS)
   Fe.addsizer(yG.PercentSizer,percents=10)
   Fe.addstrategy(strategy=cross,stop_loss=0.1,take_profit=0.08,low=14,high=90)
   print('gotem')
   Fe.Wn()
   Fe.plot()
   return
 else:
  return
@F.on('end_connection',namespace='/test')
def WY(KP,msg):
 F.disconnect(KP,namespace='/test')
 F.disconnect(KP)
 print('disconnecting...')
 F.emit('logs',"disconnecting...",namespace='/test',room=KP)
 return 'Disconnected'
KT=0
Ke=0
if __name__=='__main__':
 F.Wn(K)
