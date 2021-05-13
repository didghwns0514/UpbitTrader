if __name__ == '__main__':
	from ..Async_module import async_wrapper
else:
	try:
		from BusinessLogic.Async_module import async_wrapper
	except ImportError:
		from ..Async_module import async_wrapper
	finally:
		import hashlib
		from urllib.parse import urlencode
		import jwt
		import requests
		import uuid

import pandas as pd
import datetime
import time
import pyupbit

class Minute:

	def __init__(self): pass

	@staticmethod
	def nonasync_past_data_wrapper( _coinName, n_days=2, count=200, to=datetime.datetime.now()):
		_type = 1
		request_num = int(n_days) * 24 * 60
		#print(f'request_num//count : {request_num//count}, request_num % count : {request_num % count} ')
		iteration= range( (request_num//count) + 1 + (0 if request_num % count == 0 else 1 ))

		arg_function = [ Minute.past_data_wrapper for _ in iteration]
		arg_param = [ (_coinName, (to - datetime.timedelta(minutes=200*i)).strftime("%Y-%m-%d %H:%M:%S")) for i in iteration]
		#print(f'arg_param : {arg_param}')


		for time_value in arg_param:
			if _type == 1:
				rtn_container = async_wrapper(arg_function, arg_param)
				print(rtn_container)

			elif _type == 2:
				rtn = Minute.api_get_past_data_2(_coinName, _count=count, _to=time_value)
				print(rtn)
			elif _type == 3:
				rtn_container = async_wrapper(arg_function, arg_param)
				print(rtn_container)

	@staticmethod
	def past_data_wrapper(coin_name, n_days=2, count=200, to=datetime.datetime.now(), period=1):
		"""wrapper for get_pas_data function"""
		request_num = int(n_days) * 24 * 60
		request_cnt = request_num//count + 1
		iteration= range(request_cnt )

		arg_function = [ Minute.get_past_data for _ in iteration]
		arg_param = [[coin_name, count, (to - datetime.timedelta(minutes=200*i)).strftime("%m-%d-%Y %H:%M:%S")] for i in iteration ]

		#print(f'arg_function : {arg_function}, arg_param : {arg_param}')

		#df_container = async_wrapper(arg_function, arg_param)
		##print(f'df_container : {df_container}')
		#df_container = [df for df in df_container if isinstance(df, pd.DataFrame)]

		df_container = [Minute.get_past_data(_coinName=coin_name, _count=count, _to=_date) for _, _, _date in arg_param]
		df_container = [df for df in df_container if isinstance(df, pd.DataFrame)]

		tmp_concat = None
		if not df_container:
			return None, None
		else: # has more than one
			tmp_concat = df_container[0]
			for i, tmp_df in enumerate(df_container):
				if i >= 1:
					tmp_concat = pd.concat([tmp_concat, tmp_df])
		#print(f'tmp_concat : {tmp_concat}')
		print(f'\n'*2)
		return tmp_concat, request_cnt

	@staticmethod
	def get_past_data( _coinName, _count, _to, _period=0.1):
		""" get minute 2days data"""

		df = pyupbit.get_ohlcv(
			ticker=_coinName,
			interval="minute1",
			count=_count,
			to = _to,
			period=_period
		)

		return df

	@staticmethod
	def api_get_past_data(coin_name, _count, _to):
		#url = "https://api.upbit.com/v1/candles/minutes/1"
		print(f'coin_name : {coin_name},  _count : {_count}, _to : {_to}')
		url = "https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/1"
		queryString = {
			"market" : 'CRIX.UPBIT.'+coin_name,
			"count" : _count,
			"to" : _to
		}
		response = requests.request("GET", url, params=queryString)
		print(f'response : {response}')
		return response.json()[0]

	@staticmethod
	def api_get_past_data_2(coin_name, _count, _to):
		#url = "https://api.upbit.com/v1/candles/minutes/1"
		print(f'coin_name : {coin_name},  _count : {_count}, _to : {_to}')
		get_data_continue_candle(coin=coin_name,to=_to)
		#return response.json()[0]

def request_get(url, headers = 0, echo=0) :
	response = ""
	cnt2 = 0
	while str(response) != '<Response [200]>' and cnt2 < 10:
		if echo :
			print("requests request_get", url)
		try :
			response = requests.get(url, headers=headers)
		except Exception as e:
			print(e)
			time.sleep(20)
			cnt2 += 1
			continue
		if str(response) != '<Response [200]>':
			print('sleep ', url)
			time.sleep(15)
		cnt2 += 1
	return response.json()


# coin : "KRW-BTC"
# ex       https://crix-api-endpoint.upbit.com/v1/crix/candles/days?code=CRIX.UPBIT.KRW-BTC&count=10&to=2019-09-01%2000:00:00
def get_candle_history(coin, ty='min', interval=1, count=400, to=None) :
	ss = None
	if ty == 'day' :
		ss = 'days'
	elif ty == 'min' :
		ss = 'minutes/'+str(interval)

	base_url = 'https://crix-api-endpoint.upbit.com/v1/crix/candles/' + ss
	code = '?code=CRIX.UPBIT.' + coin
	s_count = '&count=' + str(count)

	url = base_url + code + s_count
	if to == None :
		s_to = ''
	else :
		url += ('&to='+to)
	#    print(url)
	ret = request_get(url)
	return ret

# data 중 frm보다 오래된 데이터는 지운다.
# candle : 'candleDateTime'
# tick : 'trade_time_utc'
def remove_data(data, frm, key) :
	del_items = 0
	pos = 0
	for each in data :
		if each[key] < frm :
			del_items = pos
			break
		pos += 1

	return data[:pos]

# candle from - 최근 구간 받기
def get_data_continue_candle(coin, ty='min', interval=1, count=10, frm=None, to=None) :
	end = False
	cnt = 1
	if frm != None :
		# 입력은 '2021-01-12 12:00:00'
		# 내부format 형태로 변경 2021-01-12T11:50:00+09:00
		frm = frm.replace(' ', 'T')
		frm += '+09:00'

	if to != None :
		# utc로 변경해야

		# datetime 값으로 변환
		dt_tm_kst = datetime.datetime.strptime(to,'%Y-%m-%d %H:%M:%S')
		tm_utc = dt_tm_kst - datetime.timedelta(hours=9)

		# 일자 + 시간 문자열로 변환
		to = tm_utc.strftime('%Y-%m-%d %H:%M:%S')

	next_to = to
	while(end == False) :
		t = int(time.time())
		ret = get_candle_history(coin, ty, interval, next_to)

		if len(ret) > 0 :
			print(ret[0]['candleDateTimeKst'], ret[-1]['candleDateTimeKst'])
			if len(ret) < 2 : # no more data
				return

			# candle은 내림차순
			# 마지막에 저장된 candle의 시간을 구한다.

			info = ret[-1]

			tm_kst = info['candleDateTimeKst']

			dt = info['candleDateTimeKst'].split('+')
			tm = dt[0].replace('T', ' ')
			day = tm.split(' ')[0]

			if frm != None : # from보다 이전 데이터인지 확인
				# 마지막 candle이 from보다 적으면 from 이후 candle을 지운다.
				if tm_kst < frm :
					ret = remove_data(ret, frm, 'candleDateTimeKst')
					end = True

			# 계속 검색을 하는 경우에는 현재 받은 candle의 마지막 시간이 next_to가 된다.
			# 이때 시간은  UTC
			dt = info['candleDateTime'].split('+')
			tm = dt[0].replace('T', ' ')
			next_to = tm

			# cnt 번호를 추가하여 파일이름 생성
			fname = coin+'_' + ty + '_' + str(interval) + '_' + format(cnt, '03d') + '_' + day + '.csv'
			cnt += 1
			#save_to_file_csv(fname, ret)
			print ('save ', fname, tm_kst)
			print(f'ret : {ret}')

			if ty == 'day' : # day는 400개만 받을 수 있다.ㅣ
				end = True
			else :  # 분 봉은 계속 받을 수 있다.
				time.sleep(1)
		else :
			end = True